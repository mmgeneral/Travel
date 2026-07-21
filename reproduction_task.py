
from agent import TripLeg, TransactionManager, AtomicCommitFailure, hhmm, fmt

def run_task():
    print("USER: I want to save money on the first leg of my TNN->KIX->NRT trip. Find a cheaper flight.")
    
    # 1. Setup baseline chain
    # Leg 1: TNN -> KIX (06:00 - 09:00) $300
    # Leg 2: KIX -> NRT (11:00 - 12:30) $200
    leg1_baseline = TripLeg(
        origin="TNN",
        destination="KIX",
        departure_time=hhmm("06:00"),
        arrival_time=hhmm("09:00"),
        price=300.0
    )
    leg2 = TripLeg(
        origin="KIX",
        destination="NRT",
        departure_time=hhmm("11:00"),
        arrival_time=hhmm("12:30"),
        price=200.0
    )
    
    manager = TransactionManager(chain=[leg1_baseline, leg2])
    print(f"\nAGENT: Found current itinerary:")
    print(f"  Leg 1: {leg1_baseline.origin} -> {leg1_baseline.destination} ({fmt(leg1_baseline.departure_time)} - {fmt(leg1_baseline.arrival_time)}) | ${leg1_baseline.price}")
    print(f"  Leg 2: {leg2.origin} -> {leg2.destination} ({fmt(leg2.departure_time)} - {fmt(leg2.arrival_time)}) | ${leg2.price}")
    
    # 2. Find cheaper flight (Step 1)
    print("\nAGENT: Searching for cheaper flights for Leg 1...")
    cheaper_leg1 = TripLeg(
        origin="TNN",
        destination="KIX",
        departure_time=hhmm("07:45"),
        arrival_time=hhmm("10:45"),
        price=150.0
    )
    print(f"AGENT: Found a $150 flight arriving at 10:45 AM (Leg 1).")
    
    # 3. Attempt to commit (Step 2 & 3)
    new_chain = [cheaper_leg1, leg2]
    try:
        print("\nAGENT: Attempting to update itinerary with cheaper flight...")
        manager.commit(new_chain)
    except AtomicCommitFailure as e:
        print(f"\nAGENT ERROR: {e}")
        
    # 4. Explain why (Step 4)
    print("\nAGENT RESPONSE TO USER:")
    print("I found a cheaper flight for the first leg (TNN->KIX) for only $150, but I had to reject it. ")
    print(f"Although it saves $150, it arrives at {fmt(cheaper_leg1.arrival_time)}, which is only 15 minutes before your next flight to NRT departs at {fmt(leg2.departure_time)}.")
    print("Our safety policy requires a minimum connection time (MCT) of 45 minutes for international hubs like KIX to account for potential delays and transfer procedures. ")
    print("Accepting this flight would have created a high risk of you missing your connection.")

if __name__ == "__main__":
    run_task()
