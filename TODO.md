# TODO
- Check if generators are vbetter suited for Producers and Processors, given that they could allow more efficiency through the implementation of buffer accumulation for eeither GPU batching or cpu multiprocessing.

- Refactor the offsite solution so that multiple cameras share the same processor, so extra processors should be implemented later in the graph to split the stream by camera when needed.
- Explore another GPU mode where the GPU shared time slicing values are set at deployment time depending on the number of GPU needs of the flow (in case there are less GPUs currently than what needed by the flow), and then reverted back to what it was before the run.
    - maybe, for gpu based tasks, allow the user to specify as an optional the amount of ram that task consumes in the gpu. Then let videoflow do calculations on how to distribute per physical gpu based on that task. remember that videoflow could be in the situation of not having good visibility on the number of physical gpus in the cluster, unless it takes control of the setup of the config mapping at runtime. This sohuld be a new way, added to the old way, of adding gpus (fail | share | partition)
- Explore alternatives on how to do a test run to measure the components GPU memory need and set up partitioning of GPUs based on those measurements.

- Do a code review that checks why all those imports inside functions.
