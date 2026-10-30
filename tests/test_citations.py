from eval_citations import check_answer

RANGES = {"backend/app/crud.py::authenticate": (45, 60), "backend/app/core/security.py::verify_password": (29, 32)}
SOURCES = ["backend/app/crud.py::authenticate", "backend/app/core/security.py::verify_password"]


def statuses(answer):
    return [c["status"] for c in check_answer(answer, SOURCES, RANGES)]


def test_valid_and_invalid_citations():
    assert statuses("see [1] backend/app/crud.py:45-60") == ["valid"]
    assert statuses("see [2] backend/app/utils.py:29-32") == ["wrong_path"]
    assert statuses("see [1] backend/app/crud.py:70-80") == ["wrong_lines"]
    assert statuses("see [7] backend/app/crud.py:45-60") == ["bad_index"]


def test_non_breaking_hyphen():
    assert statuses("[1] backend/app/crud.py:45\u201160") == ["valid"]
