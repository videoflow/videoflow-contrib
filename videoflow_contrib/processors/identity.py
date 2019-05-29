from videoflow.core.node import ProcessorNode

class IdentityProcessor(ProcessorNode):
    '''
    IdentityProcessor implements the identity
    function: it returns the same value that it received
    as input.
    '''
    def process(self, inp):
        return inp