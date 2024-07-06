#!/usr/bin/env python
# author      : Chenghao Mou (mouchenghao@gmail.com)
# created     : 10/4/22
from __future__ import annotations

import multiprocessing as mp
import os
import random
import re
from collections import defaultdict
from typing import Any
from typing import Callable

import click
import datasets
import numpy as np
from tqdm import tqdm

from text_dedup import logger
from text_dedup.minhash import embed_func
from text_dedup.utils import CLUSTER_COLUMN
from text_dedup.utils import INDEX_COLUMN
from text_dedup.utils import DisableReferenceCount
from text_dedup.utils import IOArgs
from text_dedup.utils import MetaArgs
from text_dedup.utils import MinHashArgs
from text_dedup.utils import Timer
from text_dedup.utils import UnionFind
from text_dedup.utils import load_hf_dataset
from text_dedup.utils import ngrams
from text_dedup.utils import optimal_param
from text_dedup.utils import sha1_hash
from text_dedup.utils import xxh3_16hash
from text_dedup.utils import xxh3_32hash

SEED = 42
RNG = np.random.RandomState(SEED)
NON_ALPHA = re.compile(r"\W", re.UNICODE)
datasets.logging.set_verbosity_error()
# for is originally used to reduce memory usage in MacOS but also ensures that the Union Find data structure
# is not copied to child processes as long as it is not modified.
mp.set_start_method("fork", force=True)
uf = UnionFind()
SIGNATURE_COLUMN = "__signatures__"


def dedup(
    column: str,
    data_path: str, 
    num_proc: int,
    ds: datasets.Dataset = None,
    batch_size: int = 10_000,
    idx_column: str = None, 
    ngram: int = 5,
    min_length: int = 5,
    num_perm: int = 250,
    threshold: float = 0.7,
    b: int | None = None,
    r: int | None = None,
    hash_func: str = "xxh3",
    hash_bits: int = 64,
):
    global uf
    uf.reset()
    HASH_BITS: int = hash_bits

    # 64 bit config is backwards compatibility mode.
    # it uses 64 bit types but almost entirely 32bit data, except for one mersenne prime 2^61
    # why legacy implementations used mersenne primes for modulo:
    # https://en.wikipedia.org/wiki/Universal_hashing#Hashing_strings
    HASH_CONFIG: dict[int, tuple[type, Any, Any]] = {
        64: (np.uint64, np.uint32((1 << 32) - 1), np.uint64((1 << 61) - 1)),
        # 32, 16 bit config does not use a mersenne prime.
        # The original reason for using mersenne prime was speed.
        # Testing reveals, there is no benefit to using a 2^61 mersenne prime for division
        32: (np.uint32, np.uint32((1 << 32) - 1), np.uint32((1 << 32) - 5)),
        16: (np.uint16, np.uint16((1 << 16) - 1), np.uint16((1 << 16) - 15)),
    }

    # defaults to backwards compatible HASH_BITS = 64, which is np.uint64 dtypes with 32bit hashes
    DTYPE, MAX_HASH, MODULO_PRIME = HASH_CONFIG.get(HASH_BITS, HASH_CONFIG[64])

    match hash_func:
        case "sha1":

            def hash_func(byte_data):
                return sha1_hash(byte_data, d=min(HASH_BITS, 32))

        case "xxh3":
            if HASH_BITS == 16:
                hash_func = xxh3_16hash
            else:
                hash_func = xxh3_32hash

    timer = Timer()

    if b is not None and r is not None:
        B, R = b, r
    else:
        # Compute the optimal `MinHashLSH` parameter that minimizes the weighted sum
        # of probabilities of false positive and false negative, taken from datasketch.
        # You can also refer to the interactive demo at https://huggingface.co/spaces/bigcode/near-deduplication.
        # The following assumes a "perfect hash". using 16 bit hashes might challenge this assumption
        # lower precision dtype will cause more collisions, so higher false_positives and less false negatives.
        # Both effects move the result towards more documents being considered duplicates.
        B, R = optimal_param(
            threshold,
            num_perm,
            false_positive_weight=0.5,
            false_negative_weight=0.5,
        )

    HASH_RANGES = [(i * R, (i + 1) * R) for i in range(B)]
    HASH_TABLES: list[dict[int, set]] = [defaultdict(set) for _ in range(B)]

    # for minhash, we need to make a lot of hashes(=num_perms).
    # In many previous implementations, this is achieved through a method described in
    # `Universal classes of hash functions` https://doi.org/10.1016/0022-0000(79)90044-8
    # There we start with a know good hash x (=hash_func) and permutate it as the following:
    # `new_hash = (a * x + b) mod prime mod max_hash` we need one a (!=0), b pair per new hash
    # the following produces these a, b pairs
    PERMUTATIONS: tuple[np.ndarray, np.ndarray] = (
        RNG.randint(
            1, MODULO_PRIME, size=(num_perm,), dtype=DTYPE
        ),  # a is a multiplier so should not be 0
        RNG.randint(0, MODULO_PRIME, size=(num_perm,), dtype=DTYPE),  # b
    )

    with timer("Total"):
        with timer("Loading"):
            if ds is None:
                ds = datasets.load_dataset('json', data_files=data_path, split='train')
            ds = ds.map(lambda x, i: {INDEX_COLUMN: i}, with_indices=True, num_proc=num_proc)
            id2id = None
            if idx_column is not None:
                original_index = ds[idx_column]
                id2id = {idx: oid for idx, oid in zip(ds[INDEX_COLUMN], original_index)}

            ds = ds.filter(
                lambda x: len(NON_ALPHA.split(x[column].lower())) >= min_length,
                num_proc=num_proc,
            )

        LEN_DATASET = len(ds)

        with timer("MinHashing"):
            embedded = ds.map(
                function=embed_func,
                fn_kwargs={
                    "num_perm": num_perm,
                    "hashranges": HASH_RANGES,
                    "ngram_size": ngram,
                    "min_length": min_length,
                    "permutations": PERMUTATIONS,
                    "hash_func": hash_func,
                    "dtype": DTYPE,
                    "max_hash": MAX_HASH,
                    "modulo_prime": MODULO_PRIME,
                },
                input_columns=[column, INDEX_COLUMN],
                remove_columns=[col for col in ds.column_names if col != INDEX_COLUMN],
                num_proc=num_proc,
                with_indices=False,
                desc="Fingerprinting...",
            )
            LEN_EMBEDDED = len(embedded)
            NUM_SHARDS = np.ceil(LEN_EMBEDDED / batch_size).astype(int)

        with timer("Clustering"):
            edges = []
            for i in tqdm(
                range(0, NUM_SHARDS),
                dynamic_ncols=True,
                desc="Iterating MinHashes...",  # noqa: E501
            ):
                embedded_shard = embedded.shard(
                    num_shards=NUM_SHARDS,
                    index=i,
                    contiguous=True,
                    writer_batch_size=batch_size,
                )
                for key, Hs in zip(embedded_shard[INDEX_COLUMN], embedded_shard[SIGNATURE_COLUMN]):
                    for i, H in enumerate(Hs):
                        HASH_TABLES[i][H].add(key)

            logger.info(f"Number of clusters: {len(HASH_TABLES)}")
            for table in tqdm(HASH_TABLES, dynamic_ncols=True, desc="Clustering..."):
                # cluster: Set[int]
                for cluster in table.values():
                    if len(cluster) <= 1:
                        continue
                    idx = min(cluster)
                    for x in cluster:
                        edges.append((x, idx))
                        uf.union(x, idx)
            logger.info(f"Number of edges: {len(set(edges))}")

        with timer("Filtering"), DisableReferenceCount():
            ds = ds.map(
                function=lambda record: {CLUSTER_COLUMN: uf.find(record[INDEX_COLUMN])},
                with_indices=False,
                num_proc=num_proc,
                new_fingerprint=str(random.getrandbits(128)),
                desc="Finding clusters...",
            )
            # This is where the deduplication happens
            # Since there is no easy groupby in datasets
            # I will use this simple filter for now
            final_data = ds.filter(
                function=lambda record: record[CLUSTER_COLUMN] == record[INDEX_COLUMN],
                with_indices=False,
                num_proc=num_proc,
                desc="Filtering clusters...",
            )

        with timer("Saving"):
            final_data = final_data.remove_columns([CLUSTER_COLUMN, INDEX_COLUMN])

    PAD = 32
    timer.report(logger=logger, pad=PAD)
    logger.info(f"{'Before':<{PAD}}: {LEN_DATASET}")
    logger.info(f"{'After':<{PAD}}: {len(final_data)}")

    return final_data

