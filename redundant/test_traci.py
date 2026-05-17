import traci

SUMO_CONFIG = "sumo/config.sumocfg"

traci.start(["sumo-gui", "-c", SUMO_CONFIG])

step = 0
breakdown_done = False   # ✅ FIX

while traci.simulation.getMinExpectedNumber() > 0:
    traci.simulationStep()

    vehicles = traci.vehicle.getIDList()

    # ✅ Breakdown logic (fixed)
    if not breakdown_done:
        if int(traci.simulation.getTime()) == 20:
            if vehicles:
                broken = vehicles[0]
                traci.vehicle.setSpeed(broken, 0)
                print(f"\n🚨 Breakdown: {broken}")
                breakdown_done = True

    print(f"\nStep {step}")
    print("Vehicles:", vehicles)

    # ✅ Simple control logic
    for v in vehicles:
        wait = traci.vehicle.getWaitingTime(v)

        if wait > 5:
            traci.vehicle.setSpeed(v, 10)

    step += 1

traci.close()