# Issue with the offside system
- add a CLAUDE.md file to both videoflow and videoflow-contrib with best practices in Python coding.
- Check if generators are vbetter suited for Producers and Processors, given that they could allow more efficiency through the implementation of buffer accumulation for eeither GPU batching or cpu multiprocessing.
- Review the code to see how to better organize videoflow core and see if there are certain functionalities that could be abstracted for better extensibility  in the future.
- Refactor the offsite solution so that multiple cameras share the same processor, so extra processors should be implemented later in the graph to split the stream by camera when needed.
