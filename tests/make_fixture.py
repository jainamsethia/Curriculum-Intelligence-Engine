"""Writes tests/fixtures/courses_small.jsonl: eleven synthetic courses of one programme (no NMIMS text), used by the API tests.
    python tests/make_fixture.py"""
import json
from pathlib import Path


def course(i, name, sem, units, prereq="NIL", outcomes=(), ay="2025-26"):
    src = "2ND YEAR/COMPUTER/SEM/synthetic.pdf"
    return {"course_name": name, "code": f"FX{i:04d}", "course_key": "".join(ch for ch in name.lower() if ch.isalnum()),
            "program": "B Tech (Computer Engineering)", "semester": sem, "academic_year": ay, "prerequisites": prereq,
            "objectives": f"This course introduces {name.lower()}.", "outcomes": [{"text": o, "k_levels": []} for o in outcomes],
            "units": [{"title": t, "text": f"{t}: {x}", "hours": 6} for t, x in units], "text_books": "", "reference_books": "", "lab_work": "",
            "source": src, "id": f"nmfx{i:06d}", "source_files": [src], "n_source_files": 1}


COURSES = [
    course(1, "Programming in C", "I", [("Basics", "variables, data types, operators and control statements in C"),
                                        ("Functions", "functions, recursion, arrays and pointers in C")], outcomes=["Write programs in C using loops and functions"]),
    course(2, "Calculus", "I", [("Limits", "limits, continuity and derivatives of functions"), ("Integrals", "definite and indefinite integrals and applications")],
           outcomes=["Apply derivatives to optimisation problems"]),
    course(3, "Digital Logic", "II", [("Boolean algebra", "logic gates, boolean algebra and Karnaugh maps"), ("Sequential circuits", "flip flops, counters and registers")]),
    course(4, "Data Structures", "III", [("Linear structures", "arrays, stacks, queues and linked lists"), ("Trees", "binary trees, binary search trees and heaps"),
                                         ("Graphs", "graph traversal and shortest paths")],
           prereq="Programming in C", outcomes=["Implement stacks, queues and linked lists", "Analyze the complexity of tree operations"]),
    course(5, "Operating Systems", "IV", [("Processes", "processes, threads and CPU scheduling"), ("Memory", "paging, segmentation and virtual memory")],
           prereq="Computer Organization and Architecture, Engineering Mathematics - I", outcomes=["Explain process scheduling algorithms"]),
    course(6, "Machine Learning", "V", [("Regression", "linear regression, logistic regression and regularisation"),
                                        ("Classification", "decision trees, support vector machines and k nearest neighbours"),
                                        ("Clustering", "k-means clustering and hierarchical clustering")],
           prereq="Calculus and Data Structures", outcomes=["Apply regression models to data", "Evaluate classifiers using cross validation"]),
    course(7, "Deep Learning", "VI", [("Neural networks", "perceptron, multilayer neural networks and backpropagation"),
                                      ("Regression", "linear regression, logistic regression and regularisation"), ("CNN", "convolutional neural networks for images")],
           prereq="Machine Learning", outcomes=["Design convolutional neural networks for image classification"]),
    course(8, "Computer Networks", "V", [("Layers", "OSI and TCP/IP models, encapsulation"), ("Routing", "distance vector and link state routing")],
           outcomes=["Explain the layers of the OSI model"]),
    course(9, "Database Systems", "IV", [("Relational model", "relations, keys, SQL queries and joins"), ("Normalisation", "functional dependencies and normal forms")],
           outcomes=["Design a relational database schema"]),
    course(10, "Software Engineering", "VI", [("Process models", "waterfall, agile and scrum process models"), ("Testing", "unit testing, integration testing and test cases")],
           prereq="Quantum Field Theory", outcomes=["Apply agile practices in a team project"]),
    course(11, "Machine Learning", "V", [("Regression", "linear regression, logistic regression and regularisation"),
                                         ("Classification", "decision trees and support vector machines")], prereq="Calculus", outcomes=["Apply regression models to data"], ay="2024-25"),
]

if __name__ == "__main__":
    out = Path(__file__).parent / "fixtures/courses_small.jsonl"
    out.write_text("".join(json.dumps(c) + "\n" for c in COURSES), encoding="utf-8")
    print(out, len(COURSES))
