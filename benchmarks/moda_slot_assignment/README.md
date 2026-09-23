# MODA slot assignment — image or text?

A shirt that was deliberately labelled "bag" came back from the Moda styling pane in the
accessory slot. This benchmark establishes why: Hopit's `/v1/outfits:rank` decides which slot
a garment occupies from the **text attributes we send at ingest**, never from the photograph.
The photograph only feeds the compatibility score between garments.

The same shirt photo is ingested under ~90 different labels (and every distinct
`(category, subcategory)` pair our own Stage 3 emits across the 22 evaluation characters),
then the ranker is asked where each copy lands. Full findings, tables and the consequences for
our pipeline are in `RESULTS.md`.

## Files here

- `run_benchmark.py` — the runner. Reads garments through the deployed app's API acting as the
  evaluation characters (same mechanism as `scripts/moda_ingest_wardrobes.py`), ingests the
  variants under a scratch `owner_ref`, runs every probe, writes `results.json`, and deletes the
  scratch owner again. Never touches our database.
- `results.json` — raw output of the run: what MODA stored per variant, every rank response
  (garment ids translated back to variant names), and the per-pair roster coverage table.
- `RESULTS.md` — the write-up.

## Running it

Needs `MODA_API` / `MODA_KEY` in `.env` (a key granting the `ingest` and `styling` scopes —
`search` and `embeddings` are not required and were not available for this run) and working
stylist logins against `CT_APP_BASE`.

```bash
python benchmarks/moda_slot_assignment/run_benchmark.py --dry-run      # list the ~230 records
python benchmarks/moda_slot_assignment/run_benchmark.py                # full run, ~10 minutes
python benchmarks/moda_slot_assignment/run_benchmark.py --skip-roster  # skip the 134 roster pairs
python benchmarks/moda_slot_assignment/run_benchmark.py --cleanup      # delete the scratch owner
```

Ingest is billed per record by Hopit, so a full run costs about 230 embeddings on their side.
The scratch owner (`ctbench_slot_assignment`) is removed in a `finally` block; `--keep` leaves it
for inspection.
