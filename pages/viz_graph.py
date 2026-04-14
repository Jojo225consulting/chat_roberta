import glob
import os

import pygwalker as pyg
import streamlit.components.v1 as components
from pygwalker.api.streamlit import StreamlitRenderer, init_streamlit_comm, get_streamlit_html
import streamlit as st
import pandas as pd
import numpy as np
import json
import base64
import requests
import time

st.set_page_config(
    layout="wide"
)

repo = st.secrets["repo"]
token = st.secrets["token"]

uploaded_file = st.file_uploader("Ajouter des données supplémentaires", type=["csv"])
# text_detail = st.text_area("Saisissez un détail pour caractériser ce fichier, les résultats/prompts, de façon unique(par exemple: prompt_v1_23_03_2026)", height=30)

rows = []

def adding_rows(path = "csv_file/*.csv"):
    data = pd.concat(
                    [
                        pd.read_csv(f, sep=",").assign(source_file=os.path.basename(f))
                        for f in glob.glob(path)
                    ],
                    ignore_index=True
                )
    return data

try:
    if st.button("Ajouter le fichier"):
        if uploaded_file is not None:
            st.write("Fichier uploadé :", uploaded_file.name)
            data = uploaded_file.read()
            # Encodage base64 (obligatoire pour GitHub API)
            content = base64.b64encode(data).decode("utf-8")
            # Infos repo
            url = f"https://api.github.com/repos/{repo}/contents/csv_file/{uploaded_file.name}"
            headers = {"Authorization": f"token {token}"}
            payload = {
                "message": "adding new csv file",
                "content": content
            }
            response = requests.put(url, headers=headers, json=payload)
            st.rerun()
                
except KeyError:
    st.write("Le format de votre fichier n'est pas le format adéquat")

#Add a 7 second delay to ensure the file is added to the repo before fetching the list of files again
time.sleep(3)

headers = {"Authorization": f"token {token}"}
url = f"https://api.github.com/repos/{repo}/contents/csv_file"
response = requests.get(url, headers=headers)

try:
    df = adding_rows()
    # init_streamlit_comm()

    vis_spec = r"""{"config":[{"config":{"defaultAggregated":true,"geoms":["line"],"coordSystem":"generic","limit":-1,"timezoneDisplayOffset":0},"encodings":{"dimensions":[{"fid":"api_key","name":"api_key","basename":"api_key","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"ID_applicant","name":"ID_applicant","basename":"ID_applicant","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"connaissances financières","name":"connaissances financières","basename":"connaissances financières","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"conscienciosité","name":"conscienciosité","basename":"conscienciosité","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"neuroticisme","name":"neuroticisme","basename":"neuroticisme","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"Commentaire CF","name":"Commentaire CF","basename":"Commentaire CF","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"Commentaire conscienciosité","name":"Commentaire conscienciosité","basename":"Commentaire conscienciosité","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"Commentaire neuroticisme","name":"Commentaire neuroticisme","basename":"Commentaire neuroticisme","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"gw_mea_key_fid","name":"Measure names","analyticType":"dimension","semanticType":"nominal"},{"fid":"détails","name":"détails","semanticType":"nominal","analyticType":"dimension","basename":"détails","dragId":"GW_DGczIEVg"}],"measures":[{"fid":"gw_count_fid","name":"Row count","analyticType":"measure","semanticType":"quantitative","aggName":"sum","computed":true,"expression":{"op":"one","params":[],"as":"gw_count_fid"}},{"fid":"gw_mea_val_fid","name":"Measure values","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"rows":[{"fid":"connaissances financières","name":"connaissances financières","basename":"connaissances financières","semanticType":"quantitative","analyticType":"dimension","offset":0}],"columns":[{"fid":"ID_applicant","name":"ID_applicant","basename":"ID_applicant","semanticType":"nominal","analyticType":"dimension","offset":0}],"color":[{"fid":"api_key","name":"api_key","basename":"api_key","semanticType":"nominal","analyticType":"dimension","offset":0}],"opacity":[],"size":[],"shape":[],"radius":[],"theta":[],"longitude":[],"latitude":[],"geoId":[],"details":[],"filters":[],"text":[]},"layout":{"showActions":false,"showTableSummary":false,"stack":"stack","interactiveScale":false,"zeroScale":true,"size":{"mode":"auto","width":320,"height":200},"format":{},"geoKey":"name","resolve":{"x":false,"y":false,"color":false,"opacity":false,"shape":false,"size":false},"renderer":"vega-lite"},"visId":"gw_mB7U","name":"Connaissances financières"},{"config":{"defaultAggregated":true,"geoms":["line"],"coordSystem":"generic","limit":-1,"timezoneDisplayOffset":0},"encodings":{"dimensions":[{"fid":"api_key","name":"api_key","basename":"api_key","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"ID_applicant","name":"ID_applicant","basename":"ID_applicant","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"connaissances financières","name":"connaissances financières","basename":"connaissances financières","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"conscienciosité","name":"conscienciosité","basename":"conscienciosité","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"neuroticisme","name":"neuroticisme","basename":"neuroticisme","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"Commentaire CF","name":"Commentaire CF","basename":"Commentaire CF","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"Commentaire conscienciosité","name":"Commentaire conscienciosité","basename":"Commentaire conscienciosité","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"Commentaire neuroticisme","name":"Commentaire neuroticisme","basename":"Commentaire neuroticisme","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"gw_mea_key_fid","name":"Measure names","analyticType":"dimension","semanticType":"nominal"},{"fid":"détails","name":"détails","semanticType":"nominal","analyticType":"dimension","basename":"détails","dragId":"GW_wG1QOyLf"}],"measures":[{"fid":"gw_count_fid","name":"Row count","analyticType":"measure","semanticType":"quantitative","aggName":"sum","computed":true,"expression":{"op":"one","params":[],"as":"gw_count_fid"}},{"fid":"gw_mea_val_fid","name":"Measure values","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"rows":[{"fid":"conscienciosité","name":"conscienciosité","basename":"conscienciosité","semanticType":"quantitative","analyticType":"dimension","offset":0}],"columns":[{"fid":"ID_applicant","name":"ID_applicant","basename":"ID_applicant","semanticType":"nominal","analyticType":"dimension","offset":0}],"color":[{"fid":"api_key","name":"api_key","basename":"api_key","semanticType":"nominal","analyticType":"dimension","offset":0}],"opacity":[],"size":[],"shape":[],"radius":[],"theta":[],"longitude":[],"latitude":[],"geoId":[],"details":[],"filters":[],"text":[]},"layout":{"showActions":false,"showTableSummary":false,"stack":"stack","interactiveScale":false,"zeroScale":true,"size":{"mode":"auto","width":320,"height":200},"format":{},"geoKey":"name","resolve":{"x":false,"y":false,"color":false,"opacity":false,"shape":false,"size":false},"renderer":"vega-lite"},"visId":"gw_bhnU","name":"Conscienciosité"},{"config":{"defaultAggregated":true,"geoms":["line"],"coordSystem":"generic","limit":-1,"timezoneDisplayOffset":0},"encodings":{"dimensions":[{"fid":"api_key","name":"api_key","basename":"api_key","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"ID_applicant","name":"ID_applicant","basename":"ID_applicant","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"connaissances financières","name":"connaissances financières","basename":"connaissances financières","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"conscienciosité","name":"conscienciosité","basename":"conscienciosité","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"neuroticisme","name":"neuroticisme","basename":"neuroticisme","semanticType":"quantitative","analyticType":"dimension","offset":0},{"fid":"Commentaire CF","name":"Commentaire CF","basename":"Commentaire CF","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"Commentaire conscienciosité","name":"Commentaire conscienciosité","basename":"Commentaire conscienciosité","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"Commentaire neuroticisme","name":"Commentaire neuroticisme","basename":"Commentaire neuroticisme","semanticType":"nominal","analyticType":"dimension","offset":0},{"fid":"gw_mea_key_fid","name":"Measure names","analyticType":"dimension","semanticType":"nominal"},{"fid":"détails","name":"détails","semanticType":"nominal","analyticType":"dimension","basename":"détails","dragId":"GW_CojJchau"}],"measures":[{"fid":"gw_count_fid","name":"Row count","analyticType":"measure","semanticType":"quantitative","aggName":"sum","computed":true,"expression":{"op":"one","params":[],"as":"gw_count_fid"}},{"fid":"gw_mea_val_fid","name":"Measure values","analyticType":"measure","semanticType":"quantitative","aggName":"sum"}],"rows":[{"fid":"neuroticisme","name":"neuroticisme","basename":"neuroticisme","semanticType":"quantitative","analyticType":"dimension","offset":0}],"columns":[{"fid":"ID_applicant","name":"ID_applicant","basename":"ID_applicant","semanticType":"nominal","analyticType":"dimension","offset":0}],"color":[{"fid":"api_key","name":"api_key","basename":"api_key","semanticType":"nominal","analyticType":"dimension","offset":0}],"opacity":[],"size":[],"shape":[],"radius":[],"theta":[],"longitude":[],"latitude":[],"geoId":[],"details":[],"filters":[],"text":[]},"layout":{"showActions":false,"showTableSummary":false,"stack":"stack","interactiveScale":false,"zeroScale":true,"size":{"mode":"auto","width":320,"height":200},"format":{},"geoKey":"name","resolve":{"x":false,"y":false,"color":false,"opacity":false,"shape":false,"size":false},"renderer":"vega-lite"},"visId":"gw_29LZ","name":"Neuroticisme"}],"chart_map":{},"workflow_list":[{"workflow":[{"type":"view","query":[{"op":"aggregate","groupBy":["ID_applicant","connaissances financières","api_key"],"measures":[]}]}]},{"workflow":[{"type":"view","query":[{"op":"aggregate","groupBy":["ID_applicant","conscienciosité","api_key"],"measures":[]}]}]},{"workflow":[{"type":"view","query":[{"op":"aggregate","groupBy":["ID_applicant","neuroticisme","api_key"],"measures":[]}]}]}],"version":"0.5.0.0"}"""

    # vis_spec_dict = json.loads(vis_spec)

    html = get_streamlit_html(df, spec=vis_spec)

    components.html(html, height=1000)
except:
    st.write("Aucun fichier csv trouvé dans le dossier csv_file du repo, ou erreur de lecture des fichiers csv")


