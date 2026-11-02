# Smart Factory Stochastic Simulator — Rebuild v1

## Model locked into this build
- Route: WS1 → WS2 → WS3 → WS4 → WS5 → WS6 → WS7 → WS9.
- WS2: MC1/MC3; WS3: MC2/MC4; WS6: MC6/MC7; WS7: MC8/MC9. Successive arrivals alternate globally; product type does not select the machine.
- WS5: MC5 only. WS4 and WS9 are manual/human resources with no mechanical failure/repair. WS1 is modeled as a kitting resource.
- Maximum WIP is user-configurable; default = 1.
- Every machine MC1–MC9 has a user-configurable input buffer; default = 0.
- Processing distributions are Normal and product-specific per resource. Missing historical machine×product combinations are explicitly marked as sibling/sparse fallbacks in `data/processing_params.csv`.
- Failure lifetime is Weibull. UI lets the user enter any two of MTTF, eta, beta and derives the third.
- Repair time is Lognormal with configurable mean and CV per machine.
- Four reliability cases are run for every simulation: No failure; Instant repair; No repair; Failure + repair.
- In No-repair, a failed machine remains unavailable. A two-machine workstation can continue on its survivor; failure of a sole required resource (e.g. MC5) can stall the order, which is reported as STALLED rather than assigned a fake makespan.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Community Cloud
1. Push the contents of this folder to a GitHub repository.
2. Open Streamlit Community Cloud and create a new app from that repository.
3. Set the main file path to `app.py`.
4. Deploy.

No secrets are required for this build.

## Data architecture
The deployed app reads compact fitted parameters from `data/processing_params.csv`; it does not ingest the hundreds of MB of raw telemetry at startup. This is deliberate. A later data-ingestion/refit layer can overwrite/update this registry from the most recent live readings while leaving the DES unchanged.

## Important provenance note
The `MC5` historical parameter source is currently the provided `WS 5, 6, 7` timing field, used as an MC5 proxy. It is labeled as such in the parameter table and should be replaced if a direct MC5 cycle-time signal is supplied.
