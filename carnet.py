from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination
from pgmpy.factors.discrete import TabularCPD

car_model = DiscreteBayesianNetwork(
    [
        ("Battery", "Radio"),
        ("Battery", "Ignition"),
        ("Ignition","Starts"),
        ("Gas","Starts"),
        ("KeyPresent", "Starts"),
        ("Starts","Moves"),
])

# Defining the parameters using CPT


cpd_battery = TabularCPD(
    variable="Battery", variable_card=2, values=[[0.70], [0.30]],
    state_names={"Battery":['Works',"Doesn't work"]},
)

cpd_gas = TabularCPD(
    variable="Gas", variable_card=2, values=[[0.40], [0.60]],
    state_names={"Gas":['Full',"Empty"]},
)

cpd_key_present = TabularCPD(
    variable="KeyPresent",
    variable_card=2,
    values=[[0.70], [0.30]],
    state_names={"KeyPresent": ["yes", "no"]},
)

cpd_radio = TabularCPD(
    variable=  "Radio", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Radio": ["turns on", "Doesn't turn on"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_ignition = TabularCPD(
    variable=  "Ignition", variable_card=2,
    values=[[0.75, 0.01],[0.25, 0.99]],
    evidence=["Battery"],
    evidence_card=[2],
    state_names={"Ignition": ["Works", "Doesn't work"],
                 "Battery": ['Works',"Doesn't work"]}
)

cpd_starts = TabularCPD(
    variable="Starts",
    variable_card=2,
    values=[
        [0.99, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01],
        [0.01, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99, 0.99],
    ],
    evidence=["Ignition", "Gas", "KeyPresent"],
    evidence_card=[2, 2, 2],
    state_names={
        "Starts": ['yes', 'no'],
        "Ignition": ["Works", "Doesn't work"],
        "Gas": ['Full', "Empty"],
        "KeyPresent": ["yes", "no"],
    },
)

cpd_moves = TabularCPD(
    variable="Moves", variable_card=2,
    values=[[0.8, 0.01],[0.2, 0.99]],
    evidence=["Starts"],
    evidence_card=[2],
    state_names={"Moves": ["yes", "no"],
                 "Starts": ['yes', 'no'] }
)

# Associating the parameters with the model structure
car_model.add_cpds(
    cpd_starts,
    cpd_ignition,
    cpd_gas,
    cpd_radio,
    cpd_battery,
    cpd_moves,
    cpd_key_present,
)

car_infer = VariableElimination(car_model)


def main() -> None:
    print("Q1: Given that the car will not move, what is the probability that the battery is not working?")
    battery_given_no_move = car_infer.query(
        variables=["Battery"], evidence={"Moves": "no"}
    )
    battery_failure_prob = battery_given_no_move.get_value(Battery="Doesn't work")
    print(f"   Answer: P(Battery=Doesn't work | Moves=no) = {battery_failure_prob:.5f}\n")

    print("Q2: Given that the radio is not working, what is the probability that the car will not start?")
    no_start_given_radio_fail = car_infer.query(
        variables=["Starts"], evidence={"Radio": "Doesn't turn on"}
    )
    prob_start_no = no_start_given_radio_fail.get_value(Starts="no")
    print(f"   Answer: P(Starts=no | Radio=Doesn't turn on) = {prob_start_no:.5f}\n")

    print(
        "Q3: Given that the battery works, does the probability of the radio working change when we also know the car has gas?"
    )
    radio_given_battery = car_infer.query(
        variables=["Radio"], evidence={"Battery": "Works"}
    )
    radio_given_battery_gas = car_infer.query(
        variables=["Radio"], evidence={"Battery": "Works", "Gas": "Full"}
    )
    prob_radio_with_battery = radio_given_battery.get_value(Radio="turns on")
    prob_radio_with_battery_gas = radio_given_battery_gas.get_value(Radio="turns on")
    delta_radio = prob_radio_with_battery_gas - prob_radio_with_battery
    print(
        f"   Answer: P(Radio=turns on | Battery=Works) = {prob_radio_with_battery:.5f}\n"
        f"           P(Radio=turns on | Battery=Works, Gas=Full) = {prob_radio_with_battery_gas:.5f}"
    )
    print(f"           Change = {delta_radio:+.5f}\n")

    print(
        "Q4: Given that the car doesn't move, how does the probability of the ignition failing change if we observe the car has no gas?"
    )
    ignition_given_no_move = car_infer.query(
        variables=["Ignition"], evidence={"Moves": "no"}
    )
    ignition_given_no_move_empty_gas = car_infer.query(
        variables=["Ignition"], evidence={"Moves": "no", "Gas": "Empty"}
    )
    prob_ignition_fail = ignition_given_no_move.get_value(Ignition="Doesn't work")
    prob_ignition_fail_empty_gas = ignition_given_no_move_empty_gas.get_value(
        Ignition="Doesn't work"
    )
    diff_ignition = prob_ignition_fail_empty_gas - prob_ignition_fail
    print(
        f"   Answer: P(Ignition=Doesn't work | Moves=no) = {prob_ignition_fail:.5f}\n"
        f"           P(Ignition=Doesn't work | Moves=no, Gas=Empty) = {prob_ignition_fail_empty_gas:.5f}"
    )
    print(f"           Change = {diff_ignition:+.5f}\n")

    print("Q5: What is the probability that the car starts if the radio works and it has gas?")
    start_given_radio_and_gas = car_infer.query(
        variables=["Starts"], evidence={"Radio": "turns on", "Gas": "Full"}
    )
    prob_start_yes = start_given_radio_and_gas.get_value(Starts="yes")
    print(f"   Answer: P(Starts=yes | Radio=turns on, Gas=Full) = {prob_start_yes:.5f}\n")

    print("Q6: Given that the car doesn't move, what is the probability that the key is not present?")
    key_absent_given_no_move = car_infer.query(
        variables=["KeyPresent"], evidence={"Moves": "no"}
    )
    prob_key_absent = key_absent_given_no_move.get_value(KeyPresent="no")
    print(f"   Answer: P(KeyPresent=no | Moves=no) = {prob_key_absent:.5f}\n")

if __name__ == "__main__":
    main()
