from datetime import datetime
from agent import TransactionManager, TripLeg, AtomicCommitFailure

def hhmm(s: str) -> datetime:
    """Helper to parse HH:MM into a datetime object for testing."""
    h, m = map(int, s.split(':'))
    return datetime(2026, 4, 19, h, m)

def run_tests():
    # 2. Define a valid 2-leg trip
    # Leg A: Arrive 12:00
    # Leg B: Depart 14:00
    leg_a = TripLeg("Origin A", "Destination A", hhmm("11:30"), hhmm("12:00"))
    leg_b = TripLeg("Destination A", "Destination B", hhmm("14:00"), hhmm("14:30"))
    
    initial_chain = [leg_a, leg_b]
    tm = TransactionManager(initial_chain, buffer_min=30)
    
    print("--- Initial State ---")
    print(f"Leg A Arrival: {leg_a.arrival_time.strftime('%H:%M')}")
    print(f"Leg B Depart:  {leg_b.departure_time.strftime('%H:%M')}")
    print(f"Buffer required: 30 min")
    
    # TEST 1 (Sunny Day): Update Leg A to arrive at 11:00. Verify COMMIT.
    print("\n--- TEST 1: Sunny Day (Update Leg A arrival to 11:00) ---")
    new_leg_a_1 = TripLeg("Origin A", "Destination A", hhmm("10:30"), hhmm("11:00"))
    test_chain_1 = [new_leg_a_1, leg_b]
    
    try:
        tm.commit(test_chain_1)
        if tm.chain[0].arrival_time == hhmm("11:00"):
            print("RESULT: PASS - Transaction committed successfully.")
        else:
            print(f"RESULT: FAIL - Chain not updated as expected. Current: {tm.chain[0].arrival_time.strftime('%H:%M')}")
    except Exception as e:
        print(f"RESULT: FAIL - Unexpected exception: {e}")

    # Reset for Test 2 to ensure we test rollback to the "original 12:00 time"
    tm = TransactionManager(initial_chain, buffer_min=30)

    # TEST 2 (Rainy Day): Update Leg A to arrive at 15:00 (after Leg B departs).
    # Verify the system triggers a ROLLBACK and keeps the original 12:00 time.
    print("\n--- TEST 2: Rainy Day (Update Leg A arrival to 15:00) ---")
    new_leg_a_2 = TripLeg("Origin A", "Destination A", hhmm("14:30"), hhmm("15:00"))
    test_chain_2 = [new_leg_a_2, leg_b]
    
    try:
        tm.commit(test_chain_2)
        print("RESULT: FAIL - Transaction committed despite temporal conflict.")
    except AtomicCommitFailure:
        if tm.chain[0].arrival_time == hhmm("12:00"):
            print("RESULT: PASS - Rollback triggered. Original 12:00 time preserved.")
        else:
            print(f"RESULT: FAIL - Data corrupted. Current arrival: {tm.chain[0].arrival_time.strftime('%H:%M')}")
    except Exception as e:
        print(f"RESULT: FAIL - Unexpected exception type: {type(e).__name__}: {e}")

if __name__ == "__main__":
    run_tests()
