class ComponentProxy(object):
    """
    Proxy object trough which communication happens. Will essentially hide
    all private methods and attributes of a component and only expose public
    functions.
    """
    __slots__ = "component"

    def __init__(self, component):
        self.component = component

    def __getattr__(self, item):
        if item.startswith("_") or not hasattr(self.component, item):
            raise AttributeError("'%s' not found on component." % item)
        return getattr(self.component, item)


class Communicator(object):
    """
    Communicator object used to exchange information and expose
    functionality between different components.
    """
    def __init__(self):
        self.__components = {}

    def __getattr__(self, item):
        if item not in self.__components:
            raise AttributeError(
                "Component '%s' not known to communicator." % item)
        return self.__components[item]

    def __str__(self):
        return "Components registered with communicator:\n\t%s" % \
            "\n\t".join("%s: %s" % (str(key), repr(value)) for key, value in
                        self.__components.iteritems())

    def register(self, component_name, component):
        """
        Register a component to the communicator.
        """
        if component_name in self.__components:
            raise ValueError("Component '%s' already registered." %
                             component_name)
        self.__components[component_name] = ComponentProxy(component)
