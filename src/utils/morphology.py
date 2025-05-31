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

    def getName(self) -> str:
        return self.__name
    
    def getInertial(self) -> Inertial:
        return self.__inertial
    
    def getVisuals(self) -> list[Visual]:
        return self.__visuals
    
    def getCollisions(self) -> list[Collision]:
        return self.__collisions
    
    def __str__(self):
        string = "Link: " + self.getName() + "\n"
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

class Robot:
    def __init__(self, name: str):
        self.__name: str = name
        self.__links: list[Link] = []
        self.__joints: list[Joint] = []

    def getName(self) -> str:
        return self.__name
    
    def getLinks(self) -> list[Link]:
        return self.__links
    
    def getJoints(self) -> list[Joint]:
        return self.__joints

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

### TODO Create a Robot from a given URDF file

def convertUrdfToRobot(filename: str):
    # create doc object
    cwd = os.getcwd()
    path = os.path.join(cwd, ASSETS, URDF, filename)
    doc = minidom.parse(path)

    # create robot object
    robot = Robot(doc.getElementsByTagName("robot")[0].getAttribute("name"))

    # get links and joints
    for linkDiv in doc.getElementsByTagName("link"):
        name = linkDiv.getAttribute("name")
        

        link = Link(name)
        robot.addLink(link)

    print(robot)

### TODO Evolve a set of Robots based on a given set and some data

def evolve(robots: list[Robot], data) -> list[Robot]:
    pass

if __name__ == "__main__":
    
    # a tree is given to mutation service to produce another tree
    # tree is fed to URDF creator
    
    # TEST 
    # FYI I just made up this design, I have no idea how it looks or if it works, Safa please give it a shot
    robot: Robot = Robot("robot_name")
    link1: Link = Link("my_link", 
                       Inertial(1, (100, 0, 0, 100, 0, 100), Origin((0, 0, 0.5), (0, 0, 0))), 
                       [Visual(Box((1, 1, 1)), 
                               None, 
                               Origin((0, 0, 0), (0, 0, 0)), 
                               Material("Cyan", (0, 1.0, 1.0, 1.0)))], 
                       [Collision(Cylinder(1, 0.5), 
                                  None, 
                                  Origin((0, 0, 0), (0, 0, 0)))])
    link2: Link = Link("another_link")
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

    convertRobotToUrdf(robot)

    convertUrdfToRobot("robot_name.urdf")
