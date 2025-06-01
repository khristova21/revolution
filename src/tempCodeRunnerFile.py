initialRobot = convertUrdfToRobot("test.urdf")
    print("Running evolve")
    mutatedRobots = evolve([(initialRobot, 1)])
    print("Converting evolved robots into to URDF")
    for robot in mutatedRobots:
        convertRobotToUrdf(robot)