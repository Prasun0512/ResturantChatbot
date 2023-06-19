import os
import PyPDF2
import xlsxwriter
import pandas as pd
import tabula
fl = os.listdir("./Docs")
for file in fl:
    mydict={}
    pdffileobj=open(os.path.join("Docs",file),'rb')
    pdfreader=PyPDF2.PdfReader(pdffileobj)
    x=len(pdfreader.pages)
    for i in range(1,x):
        page = pdfreader.pages[int(i)]
        #table_df = tabula.read_pdf(os.path.join("Docs",file), pages = x)[i]
        #print(table_df)
        mydict["page_"+str(i)]=page.extract_text().rstrip("\n").strip()        
        df = pd.DataFrame(mydict,index=[0])
        df =df.T
        df.to_csv(os.path.join("CSV",file+".csv"))          




