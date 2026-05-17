"""Hierarchical Traffic Orchestration Testing Framework (test.py).

This script serves as the unified entry point for running and testing individual
control layers, specific agents, or the entire hierarchical loop. It manages
automatic configuration generation, model loading (if trained), and live dashboard
monitoring based on clean command-line arguments.

Usage examples:
    # 1. Test a single ward agent (ward_001) in GUI mode with dashboard
    python test.py --scope ward --id ward_001

    # 2. Test an area controller (Basavanagudi) with its sibling wards
    python test.py --scope area --id Basavanagudi --num-vehicles 1500

    # 3. Test the macro city-level coordinator across all sectors
    python test.py --scope city --id Jayanagar --no-dashboard
"""

import sys
import os
import argparse
import webbrowser
from pathlib import Path

# Force the project root directory onto the system path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.ward_processor import detect_ward_boundaries
from src.traffic_generator import TrafficGenerator
from src.runtime import run_simulation
from src.topology import Topology


def main():
    parser = argparse.ArgumentParser(
        description="HMRL Traffic Orchestration Unified Testing Entrypoint",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # 1. Scope and identifiers
    parser.add_argument(
        "--scope", "-s",
        choices=["ward", "area", "city"],
        default="ward",
        help="Hierarchical scale to run/test."
    )
    parser.add_argument(
        "--id", "-i",
        default="ward_070",
        help="Identifier of the scope (e.g. ward_070, HSR_Layout, Jayanagar)."
    )
    
    # 2. Demand and simulation parameters
    parser.add_argument(
        "--num-vehicles", "-n",
        type=int,
        default=3000,
        help="Total vehicle count to generate for the simulation runs."
    )
    parser.add_argument(
        "--scenario",
        default="chaos_mode",
        help="Traffic scenario profile to spawn (normal, peak_congestion, chaos_mode)."
    )
    parser.add_argument(
        "--ticks", "-t",
        type=int,
        default=3600,
        help="Maximum simulation steps / duration in seconds."
    )
    
    # 3. Visualization and controls
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Disable the interactive SUMO GUI window."
    )
    parser.add_argument(
        "--delay", "-d",
        type=float,
        default=100.0,
        help="SUMO GUI step delay in milliseconds (makes cars watchable)."
    )
    parser.add_argument(
        "--no-dashboard",
        action="store_true",
        help="Disable starting the Flask visual dashboard server."
    )
    
    # 4. Strict road permission filtering
    parser.add_argument(
        "--no-strict-boundary",
        action="store_true",
        help="Allow spawning vehicles on non-drivable paths like footways/steps."
    )
    
    # 5. Algorithm and model loading
    parser.add_argument(
        "--algorithm", "-a",
        default="ppo",
        choices=["ppo", "dqn", "a2c"],
        help="RL algorithm folder for loading trained ward agent models."
    )

    args = parser.parse_args()

    print("=" * 80)
    print("                 HMRL Traffic System Unified Testing Console")
    print("=" * 80)
    print(f"[*] Scope:           {args.scope.upper()} ({args.id})")
    print(f"[*] Vehicles:        {args.num_vehicles:,} units")
    print(f"[*] Scenario:        {args.scenario}")
    print(f"[*] GUI Mode:        {not args.no_gui} (delay={args.delay}ms)")
    print(f"[*] Web Dashboard:   {not args.no_dashboard}")
    print(f"[*] Strict Boundary: {not args.no_strict_boundary}")
    print(f"[*] Model Directory: models/{args.algorithm}/")
    print("-" * 80)

    # Resolve primary ward based on input scope
    topology = Topology(PROJECT_ROOT)
    if args.scope == "ward":
        primary_ward = args.id
    elif args.scope == "area":
        wards = topology.get_area_wards(args.id)
        if not wards:
            print(f"[-] Error: Area '{args.id}' not found in the topology registry.")
            print(f"    Available areas: {topology.get_all_area_ids()}")
            sys.exit(1)
        primary_ward = wards[0]
    elif args.scope == "city":
        areas = topology.get_all_area_ids()
        if args.id in areas:
            primary_ward = topology.get_area_wards(args.id)[0]
        else:
            primary_ward = topology.get_area_wards(areas[0])[0]

    # Step 1: Pre-process boundaries for the simulation run
    strict_mode = not args.no_strict_boundary
    print(f"[*] Step 1: Parsing road network permissions (strict_mode={strict_mode})...")
    try:
        detect_ward_boundaries(primary_ward, PROJECT_ROOT, strict_mode=strict_mode)
    except Exception as e:
        print(f"[-] Preprocessing error: {e}")
        print("    Please ensure maps/processed/{primary_ward}/ward.net.xml exists.")
        sys.exit(1)

    # Step 2: Generate dynamic traffic routes
    print(f"[*] Step 2: Generating route file (.rou.xml) for {args.num_vehicles} cars...")
    generator = TrafficGenerator(PROJECT_ROOT, primary_ward, scenario_id=args.scenario)
    generator.generate_ward_routes(num_vehicles=args.num_vehicles)
    generator.generate_ward_sumocfg()
    print("[+] Done! Demand and configuration files compiled.")

    # Step 3: Open visual web dashboard in browser if requested
    if not args.no_dashboard:
        print("\n[*] Step 3: Launching visual interface...")
        print("    -> Visualizing layers at: http://localhost:5050")
        try:
            webbrowser.open("http://localhost:5050")
        except Exception:
            pass

    # Step 4: Run the bidirectional multi-timescale simulation
    print("\n[*] Step 4: Initializing TraCI pipeline...")
    print("    -> Press the green 'Play' button in the SUMO GUI window to start vehicle flow.")
    print("    -> Press Ctrl+C in this terminal to stop.")
    print("-" * 80)

    try:
        run_simulation(
            scope=args.scope,
            identifier=args.id,
            project_root=PROJECT_ROOT,
            gui=not args.no_gui,
            scenario_id=args.scenario,
            max_ticks=args.ticks,
            algorithm=args.algorithm,
            dashboard=not args.no_dashboard,
            gui_delay_ms=args.delay
        )
    except KeyboardInterrupt:
        print("\n[-] Testing session stopped by user.")
    except Exception as e:
        print(f"\n[-] Testing session crashed: {e}")
        print("    Make sure SUMO is installed and added to your system PATH.")


if __name__ == "__main__":
    main()
