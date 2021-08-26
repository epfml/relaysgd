import json
import os

import jinja2
import pandas as pd
import requests
import seaborn as sns
from IPython.core.display import Javascript, clear_output, display
from IPython.core.magic import register_cell_magic, register_line_magic

sns.set_style("whitegrid")

# Connection to the experimental results
GRAPHQL_URL = os.getenv("GRAPHQL_URL", //ANONYMIZED FOR DOUBLE-BLIND PEER-REVIEW//)


def query(query_string, variables={}):
    r = requests.post(GRAPHQL_URL, json={"query": query_string, "variables": variables})
    if r.status_code != 200:
        raise Exception(json.dumps(json.loads(r.text)["errors"], indent=1))
        return
    return json.loads(r.text)["data"]


def job_config(job):
    """Extract config dictionary from GraphQL result"""
    return {x["key"]: x["value"] for x in job["config"]}




templates = {}

class TemplateLoader(jinja2.BaseLoader):
    def get_source(self, environment, template):
        if template in templates:
            up_to_date_fn = lambda: False  # always reload
            return templates[template], None, up_to_date_fn
        else:
            raise jinja2.TemplateNotFound(template)

# http://eosrei.net/articles/2015/11/latex-templates-python-and-jinja2-generate-pdfs
jinja = jinja2.Environment(
    block_start_string="\Block{",
    block_end_string="}",
    variable_start_string="\Var{",
    variable_end_string="}",
    comment_start_string="\#{",
    comment_end_string="}",
    line_statement_prefix="%%",
    line_comment_prefix="%#",
    trim_blocks=True,
    autoescape=False,
    loader=TemplateLoader(),
)

def register_filter(fnc):
    jinja.filters[fnc.__name__] = fnc
    
def register_global(name, variable):
    jinja.globals[name] = variable
    
def register_test(name, test_fn):
    jinja.tests[name] = test_fn

@register_cell_magic
def template(line, cell):
    name = line
    template_string = cell
    templates[name] = template_string

def render(template_name, *args, **kwargs):
    return jinja.get_template(template_name).render(*args, **kwargs)

display(
    Javascript(
        'Jupyter.CodeCell.options_default.highlight_modes.magic_tex = { reg: ["^%%template"] };'
    )
)
