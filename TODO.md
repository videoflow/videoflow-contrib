# Issue with the offside system
The __main__ gotcha bit me (the one my own plan warned about): running the flow script directly made glue-node classes __main__.X, unimportable by workers. Fixed with a driver that imports the module — worth keeping in mind for the real offside.py deployment (it's already structured correctly for videoflow deploy, which imports build_flow).

A cosmetic videoflow-core issue (not my components): workers log a NATS-drain TimeoutError at shutdown, after all work completes — output was fully intact. If you want, I can dig into videoflow/messaging to make worker shutdown drain cleanly.

Upload models to github
