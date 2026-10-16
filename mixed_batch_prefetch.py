"""Ordered CPU batch production, optionally overlapping one batch with training.

The builder must be a pure CPU data operation using only explicit/local NumPy
generators. It must not consume Python/NumPy/Torch global RNG, touch CUDA, or
change training state. Existing mixed-source builders derive extraction seeds
from source/patch/frame identity and satisfy this requirement.

Use one context per epoch. A consumed batch's successor is submitted before
yielding the batch to the caller, so its extraction overlaps the caller's GPU
step. Results stay in the exact supplied spec order. Resume supplies the same
epoch specs and the saved next_update; any unconsumed speculative batch is
discarded at exit and rebuilt deterministically when needed.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager


@contextmanager
def batch_stream(sources, specs, seed, batch_builder, *, start_update=0, prefetch_batches=0):
    """Yield an iterator of ``(update, batch, failed)`` with zero or one future.

    Exiting the context closes its iterator and waits for the outstanding CPU
    operation to finish, including when training raises or intentionally stops.
    A consumed builder failure propagates at its original update position.
    An intentionally unconsumed speculative result/failure is discarded.
    """
    if prefetch_batches not in (0, 1):
        raise ValueError("prefetch_batches must be 0 (sequential) or 1 (one future)")
    if not 0 <= start_update <= len(specs):
        raise ValueError("start_update must identify this epoch's next update")
    executor = None
    pending = None

    def produce():
        nonlocal pending
        if prefetch_batches == 0:
            for update in range(start_update, len(specs)):
                batch, failed = batch_builder(sources, specs[update], seed)
                yield update, batch, failed
            return
        if start_update == len(specs):
            return
        pending = executor.submit(batch_builder, sources, specs[start_update], seed)
        for update in range(start_update, len(specs)):
            batch, failed = pending.result()
            pending = None
            if update + 1 < len(specs):
                pending = executor.submit(batch_builder, sources, specs[update + 1], seed)
            yield update, batch, failed

    if prefetch_batches == 1:
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mixed-batch-cpu")
    stream = produce()
    try:
        yield stream
    finally:
        stream.close()
        if pending is not None:
            pending.cancel()
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        pending = None
