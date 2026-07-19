# TODO
- add a CLAUDE.md file to both videoflow and videoflow-contrib with best practices in Python coding.
    - Include type hints for all parameters to functions and return types.
    - Include taht documentation needs to be updated after each modification.
- Check if generators are vbetter suited for Producers and Processors, given that they could allow more efficiency through the implementation of buffer accumulation for eeither GPU batching or cpu multiprocessing.
- Review the code to see how to better organize videoflow core and see if there are certain functionalities that could be abstracted for better extensibility  in the future.
- Refactor the offsite solution so that multiple cameras share the same processor, so extra processors should be implemented later in the graph to split the stream by camera when needed.
- Explore another GPU mode where the GPU shared time slicing values are set at deployment time depending on the number of GPU needs of the flow (in case there are less GPUs currently than what needed by the flow), and then reverted back to what it was before the run.
- Explore alternatives on how to do a test run to measure the components GPU memory need and set up partitioning of GPUs based on those measurements.
- Do a code review that checks why all those imports inside functions.
