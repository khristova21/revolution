from collections.abc import Iterable
from enum import Enum
from xml.dom import minidom
import os

### Constants
ASSETS = "assets"
URDF = "urdf"

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
    
    def __str__(self):
        string = "Origin: "
        if (xyz := self.getXyz()) is not None:
            string += f"x={xyz[0]} y={xyz[1]} z={xyz[2]}   "
        if (rpy := self.getRpy()) is not None:
            string += f"r={rpy[0]} p={rpy[1]} y={rpy[2]}"
        string += "\n"
        return string

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
    
    def __str__(self):
        string = "Inertial:\n"
        if (origin := self.getOrigin()) is not None:
            string += "*       " + str(origin)
        string += f"*       Mass: {self.getMass()}\n"
        inertia = self.getInertia()
        string += f"*       Inertia: ixx={inertia[0]} ixy={inertia[1]} ixz={inertia[2]} iyy={inertia[3]} iyz={inertia[4]} izz={inertia[5]}\n"
        return string
    
class Geometry():
    def __init__(self, name: str, dictionary: dict[str, any]):
        self.__name: str = name
        self.__attributes: dict[str, any] = dictionary

    def getName(self):
        return self.__name

    def getAttributes(self) -> dict[str, any]:
        return self.__attributes
    
    def __str__(self):
        string = "Geometry:\n"
        string += f"*         {self.getName().capitalize()}: "
        for key, value in self.getAttributes().items():
            string += f"{key}={value} "
        string += "\n"
        return string
    
class Box(Geometry):
    def __init__(self, size: tuple[float, float, float]):
        super().__init__("box", {'size' : (size[0], size[1], size[2])})

class Cylinder(Geometry):
    def __init__(self, radius: float, length: float):
        super().__init__("cylinder", {'radius' : radius, 'length': length})

class Capsule(Geometry):
    def __init__(self, radius: float, length: float):
        super().__init__("capsule", {'radius' : radius, 'length': length})

class Sphere(Geometry):
    def __init__(self, radius: float):
        super().__init__("sphere", {'radius' : radius})

class Mesh(Geometry):
    def __init__(self, filename: str, scale: float = None):
        super().__init__("mesh", {'filename' : filename, 'scale' : scale})

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
    
    def __str__(self):
        string = f"{self.__class__.__name__}:"
        if (name := self.getName()) is not None:
            string += " " + name
        string += "\n"
        if (origin := self.getOrigin()) is not None:
            string += "*       " + str(origin)
        string += "*       " + str(self.getGeometry())
        return string
    
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
    
    def __str__(self):
        string = "Material: " + self.getName() + "\n"
        if (color := self.getColor()) is not None:
            string += f"*         Color: r={color[0]} g={color[1]} b={color[2]} a={color[3]}   "
        if (texture := self.getTexture()) is not None:
            string += f"texture={texture}"
        string += "\n"
        return string

class Visual(Collision):
    def __init__(self, 
                 geometry: Geometry, 
                 name: str = None, 
                 origin: Origin = None, 
                 material: Material = None):
        super().__init__(geometry, name, origin)
        self.__material: Material = material

    def getMaterial(self) -> Material:
        return self.__material
    
    def __str__(self):
        string = super().__str__()
        if (material := self.getMaterial()) is not None:
            string += "*       " + str(material)
        return string

class Role(Enum):
    TORSO = "torso"
    LIMB = "limb"
    UNKNOWN = "unknown"

class Link:
    def __init__(self, 
                 name: str, 
                 inertial: Inertial = None, 
                 visuals: list[Visual] = [],
                 collisions: list[Collision] = []):
        self.__name: str = name
        self.__inertial: Inertial = inertial
        self.__visuals: list[Visual] = visuals
        self.__collisions: list[Collision] = collisions
        self.__role: Role = Link.parseName(name)

    @staticmethod
    def parseName(name: str) -> Role:
        if Role.TORSO.value in name.lower():
            return Role.TORSO
        elif Role.LIMB.value in name.lower():
            return Role.LIMB
        else:
            return Role.UNKNOWN
    
    def getName(self) -> str:
        return self.__name
    
    def getRole(self) -> Role:
        return self.__role
    
    def getInertial(self) -> Inertial:
        return self.__inertial
    
    def getVisuals(self) -> list[Visual]:
        return self.__visuals
    
    def getCollisions(self) -> list[Collision]:
        return self.__collisions
    
    def __str__(self):
        string = "Link: " + self.getName() + f"   ({self.getRole().value})"  + "\n"
        if (inertial := self.getInertial()) is not None:
            string += "*     " + str(inertial)
        for visual in self.getVisuals():
            string += "*     " + str(visual)
        for collision in self.getCollisions():
            string += "*     " + str(collision)
        return string

### Joint Attributes

class Type(Enum):
    REVOLUTE = "revolute"
    CONTINUOUS = "continuous"
    PRISMATIC = "prismatic"
    FIXED = "fixed"
    FLOATING = "floating"
    PLANAR = "planar"

class Calibration:
    def __init__(self, rising: float = None, falling: float = None):
        self.__rising: float = rising
        self.__falling: float = falling

    def getRising(self) -> float:
        return self.__rising
    
    def getFalling(self) -> float:
        return self.__falling
    
    def __str__(self):
        string = "Calibration: "
        if (rising := self.getRising()) is not None:
            string += f"rising={rising}   "
        if (falling := self.getFalling()) is not None:
            string += f"falling={falling}"
        string += "\n"
        return string
    
class Dynamics:
    def __init__(self, damping: float = None, friction: float = None):
        self.__damping: float = damping
        self.__friction: float = friction

    def getDamping(self) -> float:
        return self.__damping
    
    def getFriction(self) -> float:
        return self.__friction
    
    def __str__(self):
        string = "Dynamics: "
        if (damping := self.getDamping()) is not None:
            string += f"damping={damping}   "
        if (friction := self.getFriction()) is not None:
            string += f"friction={friction}"
        string += "\n"
        return string
    
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
    
    def __str__(self):
        string = f"Limit: "
        if (lower := self.getLower()) is not None:
            string += f"lower={lower} "
        if (upper := self.getUpper()) is not None:
            string += f"upper={upper} "
        string += f"effort={self.getEffort()} velocity={self.getVelocity()}"
        string += "\n"
        return string

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
    
    def __str__(self):
        string = f"Mimic: joint={self.getName()} "
        if (multiplier := self.getMultiplier()) is not None:
            string += f"multiplier={multiplier} "
        if (offset := self.getOffset()) is not None:
            string += f"offset={offset}"
        string += "\n"
        return string
    
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
    
    def __str__(self):
        string = "Safety Controller: "
        if (low := self.getSoftLowerLimit()) is not None:
            string += f"Soft Lower Limit={low} "
        if (upper := self.getSoftUpperLimit()) is not None:
            string += f"Soft Upper Limit={upper} "
        if (kpos := self.getKPosition()) is not None:
            string += f"K Position={kpos} "
        string += f"K Velocity={self.getKVelocity()}"
        string += "\n"
        return string

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
        
    def getName(self) -> str:
        return self.__name
    
    def getType(self) -> Type:
        return self.__type
    
    def getParent(self) -> Link:
        return self.__parent
    
    def getChild(self) -> Link:
        return self.__child
    
    def getOrigin(self) -> Origin:
        return self.__origin
    
    def getAxis(self) -> tuple[float, float, float]:
        return self.__axis
    
    def getCalibration(self) -> Calibration:
        return self.__calibration
    
    def getDynamics(self) -> Dynamics:
        return self.__dynamics
    
    def getLimit(self) -> Limit:
        return self.__limit
    
    def getMimic(self) -> Mimic:
        return self.__mimic
    
    def getSafetyController(self) -> SafetyController:
        return self.__safetyController
    
    def __str__(self):
        string = "Joint: " + self.getName() + f"   ({self.getType().value})" + "\n"
        if (origin := self.getOrigin()) is not None:
            string += "*     " + str(origin)
        string += f"*     Parent: {self.getParent().getName()}\n"
        string += f"*     Child: {self.getChild().getName()}\n"
        if (axis := self.getAxis()) is not None:
            string += f"*     Axis: x={axis[0]} y={axis[1]} z={axis[2]}"
        if (calibration := self.getCalibration()) is not None:
            string += "*     " + str(calibration)
        if (dynamics := self.getDynamics()) is not None:
            string += "*     " + str(dynamics)
        if (limit := self.getLimit()) is not None:
            string += "*     " + str(limit)
        if (mimic := self.getMimic()) is not None:
            string += "*     " + str(mimic)
        if (safetyController := self.getSafetyController()) is not None:
            string += "*     " + str(safetyController)
        return string

### Robot Attributes
# Assumes a Robot will always have a torso link.
class Robot:
    def __init__(self, name: str):
        self.__name: str = name
        self.__links: list[Link] = []
        self.__joints: list[Joint] = []

    def getName(self) -> str:
        return self.__name
    
    def getLinks(self) -> list[Link]:
        return self.__links
    
    def getLinkByName(self, name: str) -> Link:
        for link in self.getLinks():
            if link.getName() == name:
                return link
        return None
    
    def getJoints(self) -> list[Joint]:
        return self.__joints
    
    # FIXME - Return a list of links if we end up 
    # having multiple torsos in a robot.
    def getTorso(self) -> Link:
        # Return the first TORSO link that is found. 
        for link in self.getLinks():
            if link.getRole() is Role.TORSO:
                return link

    def addLink(self, link: Link):
        self.__links.append(link)

    def addJoint(self, joint: Joint):
        self.__joints.append(joint)

    def __str__(self):
        string = "> Robot: " + self.getName() + "\n"
        for link in self.getLinks():
            string += "*   " + str(link)
        for joint in self.getJoints():
            string += "*   " + str(joint)
        return string

### Create URDF file based on given Robot

def convertRobotToUrdf(robot: Robot):
    # create document
    doc = minidom.Document()

    # add robot div
    robotDiv = doc.createElement('robot')
    robotDiv.setAttribute('name', robot.getName())
    doc.appendChild(robotDiv)

    # add link divs
    for link in robot.getLinks():
        linkDiv = doc.createElement('link')
        linkDiv.setAttribute('name', link.getName())

        if (inertial := link.getInertial()) is not None:
            inertialDiv = doc.createElement('inertial')
            if (origin := inertial.getOrigin()) is not None:
                originDiv = doc.createElement('origin')
                if (xyz := origin.getXyz()) is not None:
                    originDiv.setAttribute('xyz', f'{xyz[0]} {xyz[1]} {xyz[2]}')
                if (rpy := origin.getRpy()) is not None:
                    originDiv.setAttribute('rpy', f'{rpy[0]} {rpy[1]} {rpy[2]}')
                inertialDiv.appendChild(originDiv)

            massDiv = doc.createElement('mass')
            massDiv.setAttribute('value', f'{inertial.getMass()}')
            inertialDiv.appendChild(massDiv)

            inertia = inertial.getInertia()
            inertiaDiv = doc.createElement('inertia')
            inertiaDiv.setAttribute('ixx', f'{inertia[0]}')
            inertiaDiv.setAttribute('ixy', f'{inertia[1]}')
            inertiaDiv.setAttribute('ixz', f'{inertia[2]}')
            inertiaDiv.setAttribute('iyy', f'{inertia[3]}')
            inertiaDiv.setAttribute('iyz', f'{inertia[4]}')
            inertiaDiv.setAttribute('izz', f'{inertia[5]}')
            inertialDiv.appendChild(inertiaDiv)
            linkDiv.appendChild(inertialDiv)

        for visual in link.getVisuals():
            visualDiv = doc.createElement('visual')
            if (name := visual.getName()) is not None:
                visualDiv.setAttribute('name', name)
            if (origin := visual.getOrigin()) is not None:
                originDiv = doc.createElement('origin')
                if (xyz := origin.getXyz()) is not None:
                    originDiv.setAttribute('xyz', f'{xyz[0]} {xyz[1]} {xyz[2]}')
                if (rpy := origin.getRpy()) is not None:
                    originDiv.setAttribute('rpy', f'{rpy[0]} {rpy[1]} {rpy[2]}')
                visualDiv.appendChild(originDiv)
            
            geometry = visual.getGeometry()
            geometryDiv = doc.createElement('geometry')
            shapeDiv = doc.createElement(geometry.getName())
            for attribute, value in geometry.getAttributes().items():
                if isinstance(value, Iterable):
                    string = ""
                    for v in value:
                        string += f"{v} "
                    shapeDiv.setAttribute(attribute, string.rstrip())
                else:
                    shapeDiv.setAttribute(attribute, f'{value}')
            geometryDiv.appendChild(shapeDiv)
            visualDiv.appendChild(geometryDiv)

            if (material := visual.getMaterial()) is not None:
                materialDiv = doc.createElement('material')
                materialDiv.setAttribute('name', material.getName())

                if (color := material.getColor()) is not None:
                    colorDiv = doc.createElement('color')
                    colorDiv.setAttribute('rgba', f'{color[0]} {color[1]} {color[2]} {color[3]}')
                    materialDiv.appendChild(colorDiv)
                
                if (texture := material.getTexture()) is not None:
                    textureDiv = doc.createElement('texture')
                    textureDiv.setAttribute('filename', texture)
                    materialDiv.appendChild(textureDiv)
                visualDiv.appendChild(materialDiv)
            linkDiv.appendChild(visualDiv)

        for collision in link.getCollisions():
            collisionDiv = doc.createElement('collision')
            if (name := collision.getName()) is not None:
                collisionDiv.setAttribute('name', name)
            if (origin := collision.getOrigin()) is not None:
                originDiv = doc.createElement('origin')
                if (xyz := origin.getXyz()) is not None:
                    originDiv.setAttribute('xyz', f'{xyz[0]} {xyz[1]} {xyz[2]}')
                if (rpy := origin.getRpy()) is not None:
                    originDiv.setAttribute('rpy', f'{rpy[0]} {rpy[1]} {rpy[2]}')
                collisionDiv.appendChild(originDiv)

            geometry = collision.getGeometry()
            geometryDiv = doc.createElement('geometry')
            shapeDiv = doc.createElement(geometry.getName())
            for attribute, value in geometry.getAttributes().items():
                if isinstance(value, Iterable):
                    string = ""
                    for v in value:
                        string += f"{v} "
                    shapeDiv.setAttribute(attribute, string.rstrip())
                else:
                    shapeDiv.setAttribute(attribute, f'{value}')
            geometryDiv.appendChild(shapeDiv)
            collisionDiv.appendChild(geometryDiv)
            linkDiv.appendChild(collisionDiv)
        robotDiv.appendChild(linkDiv)

    # add joint divs
    for joint in robot.getJoints():
        jointDiv = doc.createElement('joint')
        jointDiv.setAttribute('name', joint.getName())
        jointDiv.setAttribute('type', joint.getType().value)

        if (origin := joint.getOrigin()) is not None:
            originDiv = doc.createElement('origin')
            if (xyz := origin.getXyz()) is not None:
                originDiv.setAttribute('xyz', f'{xyz[0]} {xyz[1]} {xyz[2]}')
            if (rpy := origin.getRpy()) is not None:
                originDiv.setAttribute('rpy', f'{rpy[0]} {rpy[1]} {rpy[2]}')
            jointDiv.appendChild(originDiv)

        parent = joint.getParent()
        parentDiv = doc.createElement('parent')
        parentDiv.setAttribute('link', parent.getName())
        jointDiv.appendChild(parentDiv)

        child = joint.getChild()
        childDiv = doc.createElement('child')
        childDiv.setAttribute('link', child.getName())
        jointDiv.appendChild(childDiv)

        if (axis := joint.getAxis()) is not None:
            axisDiv = doc.createElement('axis')
            axisDiv.setAttribute('xyz', f'{axis[0]} {axis[1]} {axis[2]}')
            jointDiv.appendChild(axisDiv)

        if (calibration := joint.getCalibration()) is not None:
            calibrationDiv = doc.createElement('calibration')
            if (rising := calibration.getRising()) is not None:
                calibrationDiv.setAttribute('rising', f'{rising}')
            if (falling := calibration.getFalling()) is not None:
                calibrationDiv.setAttribute('falling', f'{falling}')
            jointDiv.appendChild(calibrationDiv)

        if (dynamics := joint.getDynamics()) is not None:
            dynamicsDiv = doc.createElement('dynamics')
            if (damping := dynamics.getDamping()) is not None:
                dynamicsDiv.setAttribute('damping', f'{damping}')
            if (friction := dynamics.getFriction()) is not None:
                dynamicsDiv.setAttribute('friction', f'{friction}')
            jointDiv.appendChild(dynamicsDiv)

        if (limit := joint.getLimit()) is not None:
            limitDiv = doc.createElement('limit')
            limitDiv.setAttribute('effort', f'{limit.getEffort()}')
            limitDiv.setAttribute('velocity', f'{limit.getVelocity()}')
            if (lower := limit.getLower()) is not None:
                limitDiv.setAttribute('lower', f'{lower}')
            if (upper := limit.getUpper()) is not None:
                limitDiv.setAttribute('upper', f'{upper}')
            jointDiv.appendChild(limitDiv)
        
        if (mimic := joint.getMimic()) is not None:
            mimicDiv = doc.createElement('mimic')
            mimicDiv.setAttribute('joint', mimic.getName())
            if (multiplier := mimic.getMultiplier()) is not None:
                mimicDiv.setAttribute('multiplier', f'{multiplier}')
            if (offset := mimic.getOffset()) is not None:
                mimicDiv.setAttribute('offset', f'{offset}')
            jointDiv.appendChild(mimicDiv)

        if (safetyController := joint.getSafetyController()) is not None:
            safetyControllerDiv = doc.createElement('safety_controller')
            safetyControllerDiv.setAttribute('k_velocity', f'{safetyController.getKVelocity()}')
            if (kPosition := safetyController.getKPosition()) is not None:
                safetyControllerDiv.setAttribute('k_position', f'{kPosition}')
            if (softLowerLimit := safetyController.getSoftLowerLimit()) is not None:
                safetyControllerDiv.setAttribute('soft_lower_limit', f'{softLowerLimit}')
            if (softUpperLimit := safetyController.getSoftUpperLimit()) is not None:
                safetyControllerDiv.setAttribute('soft_upper_limit', f'{softUpperLimit}')
            jointDiv.appendChild(safetyControllerDiv)

        robotDiv.appendChild(jointDiv)

    # define urdf path
    doc_str = doc.toprettyxml(indent = "\t")
    cwd = os.getcwd()
    path = os.path.join(cwd, ASSETS, URDF, f'{robot.getName()}.urdf')

    # write to file
    with open(path, "w") as f:
        f.write(doc_str)

### Create a Robot from a given URDF file

def convertUrdfToRobot(filename: str) -> Robot:
    # create doc object
    cwd = os.getcwd()
    path = os.path.join(cwd, ASSETS, URDF, filename)
    doc = minidom.parse(path)

    # create robot object
    robot = Robot(doc.getElementsByTagName("robot")[0].getAttribute("name"))

    # get links
    for linkDiv in doc.getElementsByTagName("link"):
        name = linkDiv.getAttribute("name")
        inertial = None
        visuals: list[Visual] = []
        collisions: list[Collision] = []

        for child in linkDiv.childNodes:
            if child.nodeType == minidom.Node.ELEMENT_NODE:
                if child.tagName == "inertial":
                    mass = None
                    inertia = None
                    origin = None
                    for c in child.childNodes:
                        if c.nodeType == minidom.Node.ELEMENT_NODE:
                            if c.tagName == "origin":
                                xyz = None
                                rpy = None
                                for key, value in c.attributes.items():
                                    values = value.split()
                                    if key == "xyz":
                                        xyz = (float(values[0]), float(values[1]), float(values[2]))
                                    elif key == "rpy":
                                        rpy = (float(values[0]), float(values[1]), float(values[2]))
                                origin = Origin(xyz, rpy)
                            elif c.tagName == "mass":
                                mass = c.getAttribute("value")
                            elif c.tagName == "inertia":
                                inertia = (float(c.getAttribute("ixx")),
                                           float(c.getAttribute("ixy")),
                                           float(c.getAttribute("ixz")),
                                           float(c.getAttribute("iyy")),
                                           float(c.getAttribute("iyz")),
                                           float(c.getAttribute("izz")))
                    inertial = Inertial(mass, inertia, origin)
                elif child.tagName == "visual":
                    geometry = None
                    visualName = None
                    origin = None
                    material = None
                    if child.getAttribute("name"):
                        visualName = child.getAttribute("name")
                    for c in child.childNodes:
                        if c.nodeType == minidom.Node.ELEMENT_NODE:
                            if c.tagName == "geometry":
                                if c.childNodes[1].tagName == "box":
                                    size = c.childNodes[1].getAttribute("size").split()
                                    geometry = Box((float(size[0]), float(size[1]), float(size[2])))
                                elif c.childNodes[1].tagName == "cylinder":
                                    radius = float(c.childNodes[1].getAttribute("radius"))
                                    length = float(c.childNodes[1].getAttribute("length"))
                                    geometry = Cylinder(radius, length)
                                elif c.childNodes[1].tagName == "capsule":
                                    radius = float(c.childNodes[1].getAttribute("radius"))
                                    length = float(c.childNodes[1].getAttribute("length"))
                                    geometry = Capsule(radius, length)
                                elif c.childNodes[1].tagName == "sphere":
                                    radius = float(c.childNodes[1].getAttribute("radius"))
                                    geometry = Sphere(radius)
                                elif c.childNodes[1].tagName == "Mesh":
                                    filename = c.childNodes[1].getAttribute("filename")
                                    scale = None
                                    if c.childNodes[1].getAttribute("scale"):
                                        scale = float(c.childNodes[1].getAttribute("scale"))
                                    geometry = Mesh(filename, scale)
                            elif c.tagName == "origin":
                                xyz = None
                                rpy = None
                                for key, value in c.attributes.items():
                                    values = value.split()
                                    if key == "xyz":
                                        xyz = (float(values[0]), float(values[1]), float(values[2]))
                                    elif key == "rpy":
                                        rpy = (float(values[0]), float(values[1]), float(values[2]))
                                origin = Origin(xyz, rpy)
                            elif c.tagName == "material":
                                materialName = c.getAttribute("name")
                                color = None
                                texture = None
                                for materialChild in c.childNodes:
                                    if materialChild.nodeType == minidom.Node.ELEMENT_NODE:
                                        if materialChild.tagName == "color":
                                            rgba = materialChild.getAttribute("rgba").split()
                                            color = (float(rgba[0]), float(rgba[1]), float(rgba[2]), float(rgba[3]))
                                        elif materialChild.tagName == "texture": 
                                            texture = materialChild.getAttribute("filename")
                                material = Material(materialName, color, texture)
                    visuals.append(Visual(geometry, visualName, origin, material))
                elif child.tagName == "collision":
                    geometry = None
                    collisionName = None
                    origin = None
                    if child.getAttribute("name"):
                        collisionName = child.getAttribute("name")
                    for c in child.childNodes:
                        if c.nodeType == minidom.Node.ELEMENT_NODE:
                            if c.tagName == "geometry":
                                if c.childNodes[1].tagName == "box":
                                    size = c.childNodes[1].getAttribute("size").split()
                                    geometry = Box((float(size[0]), float(size[1]), float(size[2])))
                                elif c.childNodes[1].tagName == "cylinder":
                                    radius = float(c.childNodes[1].getAttribute("radius"))
                                    length = float(c.childNodes[1].getAttribute("length"))
                                    geometry = Cylinder(radius, length)
                                elif c.childNodes[1].tagName == "capsule":
                                    radius = float(c.childNodes[1].getAttribute("radius"))
                                    length = float(c.childNodes[1].getAttribute("length"))
                                    geometry = Capsule(radius, length)
                                elif c.childNodes[1].tagName == "sphere":
                                    radius = float(c.childNodes[1].getAttribute("radius"))
                                    geometry = Sphere(radius)
                                elif c.childNodes[1].tagName == "Mesh":
                                    filename = c.childNodes[1].getAttribute("filename")
                                    scale = None
                                    if c.childNodes[1].getAttribute("scale"):
                                        scale = float(c.childNodes[1].getAttribute("scale"))
                                    geometry = Mesh(filename, scale)
                            elif c.tagName == "origin":
                                xyz = None
                                rpy = None
                                for key, value in c.attributes.items():
                                    values = value.split()
                                    if key == "xyz":
                                        xyz = (float(values[0]), float(values[1]), float(values[2]))
                                    elif key == "rpy":
                                        rpy = (float(values[0]), float(values[1]), float(values[2]))
                                origin = Origin(xyz, rpy)
                    collisions.append(Collision(geometry, collisionName, origin))
        link = Link(name, inertial, visuals, collisions)
        robot.addLink(link)

    # get joints
    for jointDiv in doc.getElementsByTagName("joint"):
        name = jointDiv.getAttribute("name")
        jointType = Type(jointDiv.getAttribute("type"))
        parent = None
        jointChild = None
        origin = None
        axis = None
        calibration = None
        dynamics = None
        limit = None
        mimic = None
        safetyController = None

        for child in jointDiv.childNodes:
            if child.nodeType == minidom.Node.ELEMENT_NODE:
                if child.tagName == "parent":
                    parent = robot.getLinkByName(child.getAttribute("link"))
                elif child.tagName == "child":
                    jointChild = robot.getLinkByName(child.getAttribute("link"))
                elif child.tagName == "origin":
                    xyz = None
                    rpy = None
                    for key, value in child.attributes.items():
                        values = value.split()
                        if key == "xyz":
                            xyz = (float(values[0]), float(values[1]), float(values[2]))
                        elif key == "rpy":
                            rpy = (float(values[0]), float(values[1]), float(values[2]))
                    origin = Origin(xyz, rpy)
                elif child.tagName == "axis":
                    xyz = child.getAttribute("xyz").split()
                    axis = (float(xyz[0]), float(xyz[1]), float(xyz[2]))
                elif child.tagName == "calibration":
                    rising = None
                    falling = None
                    for key, value in child.attributes.items():
                        if key == "rising":
                            rising = float(value)
                        elif key == "falling":
                            falling = float(value)
                    calibration = Calibration(rising, falling)
                elif child.tagName == "dynamics":
                    damping = None
                    friction = None
                    for key, value in child.attributes.items():
                        if key == "damping":
                            damping = float(value)
                        elif key == "friction":
                            friction = float(value)
                    dynamics = Dynamics(damping, friction)
                elif child.tagName == "limit":
                    effort = None
                    velocity = None
                    lower = None
                    upper = None
                    for key, value in child.attributes.items():
                        if key == "effort":
                            effort = float(value)
                        elif key == "velocity":
                            velocity = float(value)
                        elif key == "lower":
                            lower = float(value)
                        elif key == "upper":
                            upper = float(value)
                    limit = Limit(effort, velocity, lower, upper)
                elif child.tagName == "mimic":
                    name = None
                    multiplier = None
                    offset = None
                    for key, value in child.attributes.items():
                        if key == "joint":
                            name = value
                        elif key == "multiplier":
                            multiplier = float(value)
                        elif key == "offset":
                            offset = float(value)
                    mimic = Mimic(name, multiplier, offset)
                elif child.tagName == "safety_controller":
                    kVelocity = None
                    softLowerLimit = None
                    softUpperLimit = None
                    kPosition = None
                    for key, value in child.attributes.items():
                        if key == "k_velocity":
                            kVelocity = float(value)
                        elif key == "soft_lower_limit":
                            softLowerLimit = float(value)
                        elif key == "soft_upper_limit":
                            softUpperLimit = float(value)
                        elif key == "k_position":
                            kPosition = float(value)
                    safetyController = SafetyController(kVelocity, softLowerLimit, softUpperLimit, kPosition)
        joint = Joint(name, jointType, parent, jointChild, origin, axis, calibration, dynamics, limit, mimic, safetyController)
        robot.addJoint(joint)

    return robot

if __name__ == "__main__":
    
    # a tree is given to mutation service to produce another tree
    # tree is fed to URDF creator
    
    # TEST 
    # FYI I just made up this design, I have no idea how it looks or if it works, Safa please give it a shot
    robot: Robot = Robot("robot_name")
    link1: Link = Link("torso", 
                       Inertial(1, (100, 0, 0, 100, 0, 100), Origin((0, 0, 0.5), (0, 0, 0))), 
                       [Visual(Box((1, 1, 1)), 
                               None, 
                               Origin((0, 0, 0), (0, 0, 0)), 
                               Material("Cyan", (0, 1.0, 1.0, 1.0)))], 
                       [Collision(Cylinder(1, 0.5), 
                                  None, 
                                  Origin((0, 0, 0), (0, 0, 0)))])
    link2: Link = Link("limb")
    joint: Joint = Joint("my_joint", 
                         Type.FLOATING, 
                         link1, 
                         link2,
                         Origin((0, 0, 1), (0, 0, 3.1416)),
                         None,
                         Calibration(0.0),
                         Dynamics(0.0, 0.0),
                         Limit(30, 1.0, -2.2, 0.7),
                         None,
                         SafetyController(10, -2.0, 0.5, 15))

    robot.addLink(link1)
    robot.addLink(link2)
    robot.addJoint(joint)

    print("Before:")
    print(robot)

    convertRobotToUrdf(robot)

    robot_copy = convertUrdfToRobot("robot_name.urdf")

    print("After:")
    print(robot_copy)
