from retriever import BM25, is_test, tokenize


def test_tokenize_splits_identifiers():
    tokens = tokenize("get_current_user createAccessToken")
    for t in ("get", "current", "user", "create", "access", "token"):
        assert t in tokens


def test_tokenize_drops_stopwords_and_stems():
    tokens = tokenize("How are emails sent?")
    assert "how" not in tokens and "are" not in tokens
    assert "email" in tokens


def test_is_test():
    assert is_test("backend/tests/api/test_login.py")
    assert is_test("conftest.py")
    assert not is_test("backend/app/crud.py")


def test_bm25_ranks_matching_doc_first():
    docs = [tokenize("send_email smtp message"), tokenize("delete item owner"), tokenize("login password token")]
    scores = BM25(docs).scores(tokenize("how is email sent over smtp"))
    assert scores.index(max(scores)) == 0
