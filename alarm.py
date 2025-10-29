from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

alarm_model = DiscreteBayesianNetwork(
    [
        ("Burglary", "Alarm"),
        ("Earthquake", "Alarm"),
        ("Alarm", "JohnCalls"),
        ("Alarm", "MaryCalls"),
    ]
)

# Defining the parameters using CPT
from pgmpy.factors.discrete import TabularCPD

cpd_burglary = TabularCPD(
    variable="Burglary", variable_card=2, values=[[0.999], [0.001]],
    state_names={"Burglary":['no','yes']},
)
cpd_earthquake = TabularCPD(
    variable="Earthquake", variable_card=2, values=[[0.998], [0.002]],
    state_names={"Earthquake":["no","yes"]},
)
cpd_alarm = TabularCPD(
    variable="Alarm",
    variable_card=2,
    values=[[0.999, 0.71, 0.06, 0.05], [0.001, 0.29, 0.94, 0.95]],
    evidence=["Burglary", "Earthquake"],
    evidence_card=[2, 2],
    state_names={"Burglary":['no','yes'], "Earthquake":['no','yes'], 'Alarm':['no','yes']},
)
cpd_johncalls = TabularCPD(
    variable="JohnCalls",
    variable_card=2,
    values=[[0.95, 0.1], [0.05, 0.9]],
    evidence=["Alarm"],
    evidence_card=[2],
    state_names={"Alarm":['no','yes'], "JohnCalls":['no', 'yes']},
)
cpd_marycalls = TabularCPD(
    variable="MaryCalls",
    variable_card=2,
    values=[[0.99, 0.3], [0.01, 0.7]],
    evidence=["Alarm"],
    evidence_card=[2],
    state_names={"Alarm":['no','yes'], "MaryCalls":['no', 'yes']},
)

# Associating the parameters with the model structure
alarm_model.add_cpds(
    cpd_burglary, cpd_earthquake, cpd_alarm, cpd_johncalls, cpd_marycalls)

alarm_infer = VariableElimination(alarm_model)

#print(alarm_infer.query(variables=["JohnCalls"],evidence={"Earthquake":"yes"}))
#
def main():
    print("Q1: The probability of Mary calling given that John called")
    mary_given_john = alarm_infer.query(
        variables=["MaryCalls"], evidence={"JohnCalls": "yes"}
    )
    print(
        f"   Answer: P(MaryCalls=yes | JohnCalls=yes) = "
        f"{mary_given_john.get_value(MaryCalls='yes'):.5f}\n"
    )

    print("Q2: The probability of both John and Mary calling given Alarm")
    john_and_mary_given_alarm = alarm_infer.query(
        variables=["JohnCalls", "MaryCalls"], evidence={"Alarm": "yes"}
    )
    print(
        f"   Answer: P(JohnCalls=yes, MaryCalls=yes | Alarm=yes) = "
        f"{john_and_mary_given_alarm.get_value(JohnCalls='yes', MaryCalls='yes'):.5f}\n"
    )

    print("Q3: The probability of Alarm, given that Mary called")
    alarm_given_mary = alarm_infer.query(
        variables=["Alarm"], evidence={"MaryCalls": "yes"}
    )
    print(
        f"   Answer: P(Alarm=yes | MaryCalls=yes) = "
        f"{alarm_given_mary.get_value(Alarm='yes'):.5f}\n"
    )


if __name__ == "__main__":
    main()
