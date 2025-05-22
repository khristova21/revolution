from enum import Enum
from xml.dom import minidom
import os

### Link Attributes

class Origin:
    def __init__(self, 
                 xyz: tuple[float, float, float] = None,
                 rpy: tuple[float, float, float] = None):
        self.__xyz: tuple[float, float, float] = xyz
        self.__rpy: tuple[float, float, float] = rpy

    def getXyz(self) -> tuple[float, float, float]:
        return self.__xyz
    
    def getRpy(self) -> tuple[float, float, float]:
        return self.__rpy

class Inertial:
    def __init__(self, 
                 mass: float,
                 inertia: tuple[float, float, float, float, float, float],
                 origin: Origin = None):
        self.__mass: float = mass
        self.__inertia: tuple[float, float, float, float, float, float] = inertia
        self.__origin: Origin = origin

    def getMass(self) -> float:
        return self.__mass
    
    def getInertia(self) -> tuple[float, float, float, float, float, float]:
        return self.__inertia
    
    def getOrigin(self) -> Origin:
        return self.__origin
    
class Geometry():
    def __init__(self):
        raise Exception("Sorry, this is just an interface, please use a child")

    def getAttributes(self) -> dict[str, any]:
        return self.__attributes
    
class Box(Geometry):
    def __init__(self, size: tuple[float, float, float]):
        self.__attributes: dict[str, any] = {"size": (size[0], size[1], size[2])}

class Cylinder(Geometry):
    def __init__(self, radius: float, length: float):
        self.__attributes: dict[str, any] = {"radius": radius, "length": length}

class Sphere(Geometry):
    def __init__(self, radius: float):
        self.__attributes: dict[str, any] = {"radius": radius}

class Mesh(Geometry):
    def __init__(self, filename: str, scale: float = None):
        self.__attributes: dict[str, any] = {"filename": filename, "scale": scale}

class Collision:
    def __init__(self, 
                 geometry: Geometry, 
                 name: str = None, 
                 origin: Origin = None
                 ):
        self.__geometry: Geometry = geometry
        self.__name: str = name
        self.__origin: Origin = origin

    def getGeometry(self) -> Geometry:
        return self.__geometry
    
    def getName(self) -> str:
        return self.__name
    
    def getOrigin(self) -> Origin:
        return self.__origin
    
class Material():
    def __init__(self, 
                 name: str, 
                 color: tuple[float, float, float, float] = None, 
                 texture: str = None):
        self.__name: str = name
        self.__color: tuple[float, float, float, float] = color
        self.__texture: str = texture

    def getName(self) -> str:
        return self.__name
    
    def getColor(self) -> tuple[float, float, float, float]:
        return self.__color
    
    def getTexture(self) -> str:
        return self.__texture

class Visual(Collision):
    def __init__(self, material: Material = None):
        super.__init__()
        self.__material: Material = material

    def getMaterial(self) -> Material:
        return self.__material

class Link:
    def __init__(self, 
                 name: str, 
                 inertial: Inertial = None, 
                 visuals: list[Visual] = [],
                 collisions: list[Collision] = []):
        self.__name: str = name
        self.__inertial: Inertial = inertial
        self.__visuals[Visual] = visuals
        self.__collisions[Collision] = collisions

    def getName(self) -> str:
        return self.__name
    
    def getInertial(self) -> Inertial:
        return self.__inertial
    
    def getVisuals(self) -> list[Visual]:
        return self.__visuals
    
    def getCollisions(self) -> list[Collision]:
        return self.__collisions

### Joint Attributes

class Type(Enum):
    REVOLUTE = 1
    CONTINUOUS = 2
    PRISMATIC = 3
    FIXED = 4
    FLOATING = 5
    PLANAR = 6

class Calibration:
    def __init__(self, rising: float = None, falling: float = None):
        self.__rising: float = rising
        self.__falling: float = falling

    def getRising(self) -> float:
        return self.__rising
    
    def getFalling(self) -> float:
        return self.__falling
    
class Dynamics:
    def __init__(self, damping: float = None, friction: float = None):
        self.__damping: float = damping
        self.__friction: float = friction

    def getDamping(self) -> float:
        return self.__damping
    
    def getFriction(self) -> float:
        return self.__friction
    
class Limit:
    def __init__(self, 
                 effort: float, 
                 velocity: float, 
                 lower: float = None, 
                 upper: float = None):
        self.__effort: float = effort
        self.__velocity: float = velocity
        self.__lower: float = lower
        self.__upper: float = upper

    def getEffort(self) -> float:
        return self.__effort
    
    def getVelocity(self) -> float:
        return self.__velocity
    
    def getLower(self) -> float:
        return self.__lower
    
    def getUpper(self) -> float:
        return self.__upper

class Mimic:
    def __init__(self, name: str, multiplier: float = None, offset: float = None):
        self.__name: str = name
        self.__multiplier: float = multiplier
        self.__offset: float = offset

    def getName(self) -> str:
        return self.__name
    
    def getMultiplier(self) -> float:
        return self.__multiplier
    
    def getOffset(self) -> float:
        return self.__offset
    
class SafetyController:
    def __init__(self, 
                 kVelocity: float, 
                 softLowerLimit: float = None,
                 softUpperLimit: float = None,
                 kPosition: float = None):
        self.__kVelocity: float = kVelocity
        self.__softLowerLimit: float = softLowerLimit
        self.__softUpperLimit: float = softUpperLimit
        self.__kPosition: float = kPosition

    def getKVelocity(self) -> float:
        return self.__kVelocity
    
    def getSoftLowerLimit(self) -> float:
        return self.__softLowerLimit
    
    def getSoftUpperLimit(self) -> float:
        return self.__softUpperLimit
    
    def getKPosition(self) -> float:
        return self.__kPosition

class Joint:
    def __init__(self, 
                 name: str, 
                 type: Type,
                 parent: Link,
                 child: Link,
                 origin: Origin = None,
                 axis: tuple[float, float, float] = None,
                 calibration: Calibration = None,
                 dynamics: Dynamics = None,
                 limit: Limit = None,
                 mimic: Mimic = None,
                 safetyController: SafetyController = None):
        self.__name: str = name
        self.__type: Type = type
        self.__parent: Link = parent
        self.__child: Link = child
        self.__origin: Origin = origin
        self.__axis: tuple[float, float, float] = axis
        self.__calibration: Calibration = calibration
        self.__dynamics: Dynamics = dynamics
        self.__limit: Limit = limit
        self.__mimic: Mimic = mimic
        self.__safetyController: SafetyController = safetyController

        if (type == Type.REVOLUTE or type == Type.PRISMATIC) and limit == None:
            raise ValueError("Limit is required for revolute and prismatic joints")

### Robot Attributes

class Robot:
    def __init__(self, name: str):
        self.__name: str = name
        self.__links[Link] = []
        self.__joints[Joint] = []

    def getName(self) -> str:
        return self.__name

    def addLink(self, link: Link):
        self.__links.append(link)

    def addJoint(self, joint: Joint):
        self.__joints.append(joint)

### Create URDF file based on given Robot

def createURDF():
    # create document
    doc = minidom.Document()

    # create robot
    robot = doc.createElement('robot')
    robot.setAttribute('name', 'robot')
    doc.appendChild(robot)

    # create example link
    link = doc.createElement('link')
    link.setAttribute('name', 'base_footprint')
    robot.appendChild(link)

    # define urdf path
    doc_str = doc.toprettyxml(indent = "\t")
    cwd = os.getcwd()
    path = os.path.join(cwd, "assets", "urdf", "test1.urdf")

    # write to file
    with open(path, "w") as f:
        f.write(doc_str)

if __name__ == "__main__":
    
    # a tree is given to mutation service to produce another tree
    # tree is fed to URDF creator
    
    
    # TEST
    createURDF()
