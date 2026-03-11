# Map folder name to binary class (0 = No Cancer, 1 = Cancer)
def get_binary_label(class_name):
    cancer_classes = {
        "adenocarcinoma", "large.cell.carcinoma",
        "squamous.cell.carcinoma", "malignant"
    }
    return 1 if class_name in cancer_classes else 0
