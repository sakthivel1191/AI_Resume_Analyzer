import json

golden_set = {
    "resume_01": [
        "Python",
        "SQL",
        "Machine Learning"
    ],
    "resume_02": [
        "Java",
        "HTML",
        "CSS",
        "JavaScript"
    ],
    "resume_03": [
        "Python",
        "Django",
        "MySQL"
    ],
    "resume_04": [
        "Excel",
        "SQL",
        "Power BI"
    ],
    "resume_05": [
        "C",
        "C++",
        "Python"
    ],
    "resume_06": [
        "HTML",
        "CSS",
        "React",
        "JavaScript"
    ],
    "resume_07": [
        "Python",
        "SQL",
        "Data Analysis"
    ],
    "resume_08": [
        "Java",
        "Spring Boot",
        "MySQL"
    ]
}
def calculate_metrics(expected, predicted):
    expected = set(skill.lower() for skill in expected)
    predicted = set(skill.lower() for skill in predicted)
    correct = expected.intersection(predicted)
    if len(predicted) == 0:
        precision = 0
    else:
        precision = len(correct) / len(predicted)
    if len(expected) == 0:
        recall = 0
    else:
        recall = len(correct) / len(expected)
    return precision, recall
def evaluate():
    total_precision = 0
    total_recall = 0
    count = 0
    for resume_name, expected in golden_set.items():
        output_file = ("output/" + resume_name + ".json")
        try:
            with open(output_file,"r",encoding="utf-8") as file:
                result = json.load(file)
            predicted = result.get("skills", [])
            precision, recall = calculate_metrics(expected,predicted)
            total_precision += precision
            total_recall += recall
            count += 1
            print("\nResume:", resume_name)
            print("Precision:",round(precision, 2))
            print("Recall:",round(recall, 2))
        except FileNotFoundError:
            print("Output not found:",resume_name)
    if count > 0:
        print("\nFINAL EVALUATION")
        print(
            "Average Precision:",
            round(total_precision / count, 2)
        )
        print(
            "Average Recall:",
            round(total_recall / count, 2)
        )

if __name__ == "__main__":
    evaluate()
