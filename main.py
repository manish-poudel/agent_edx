from agents.sec_filing_agent import SecFilingAgent



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    sec_filing_agent = SecFilingAgent()
    print(sec_filing_agent.invoke("Stockholder votes information of 2017"))

