
from data.company_data_faiss import CompanyDataFaiss


def find_cik(company_name: str) -> str:
    """find cik (CIK (Central Index Key)) from the company name

    Args:
        company_name: Company name whose cik is to be found. For eg, Amazon, Google
    """
    company_data_faiss = CompanyDataFaiss()
    company, cik = company_data_faiss.query(search_query=company_name)
    return f"{company_name}: {cik}"

