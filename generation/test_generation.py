import pytest
from generation import dummy_generation



# Step 1: Test for ChromaDB connection
def test_connect_chromadb():
    client, collections = dummy_generation.connect_chromadb()
    assert client is not None, "Client should not be None"
    assert isinstance(collections, list), "Collections should be a list"
    print("test_connect_chromadb: PASSED")


# Step 2: Test for listing unique companies
def test_list_unique_companies():
    _, collections = dummy_generation.connect_chromadb()
    if not collections:
        print("No collections found. Skipping test_list_unique_companies.")
        return
    collection = collections[0]
    companies = dummy_generation.list_unique_companies(collection)
    assert isinstance(companies, set), "Should return a set of companies"
    print("test_list_unique_companies: PASSED")
