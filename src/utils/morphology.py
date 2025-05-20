from xml.dom import minidom
import os

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
    path = "assets/urdf/test1.urdf"

    # write to file
    with open(path, "w") as f:
        f.write(doc_str)

if __name__ == "__main__":
    # TEST
    createURDF()