class IterationManager(object):
    """
    Class managing the different iterations.
    """
    def __init__(self, project, path):
        """
        :param project: The project instance serving as the mediator.
        :param path: The path of the iterations.
        """
        self._project = project
