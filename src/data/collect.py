"""
PhishTrace — Data Collection Pipeline
======================================
Documents how the raw dataset was assembled and provides functions
to refresh the phishing feed from live sources.

Data sources (all in data/raw/):
  dataset_phishing.csv      — merged phishing URLs (Kaggle + PhishTank)
  openphish_feed.txt        — OpenPhish live feed (phishing only)
  tranco_top10k.csv         — Tranco top-10k legitimate domains
  verified_online.csv       — PhishTank verified online URLs

Run
---
  python src/data/collect.py                 # shows dataset stats
  python src/data/collect.py --refresh       # downloads fresh PhishTank feed

Output
------
  data/processed/phishtrace_dataset.csv     — merged, labelled, deduplicated
"""

import os
import sys
import argparse
import gzip
import json
import time
import requests
import pandas as pd
from pathlib import Path
from urllib.parse import urlparse

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent.parent
RAW_DIR      = ROOT / "data" / "raw"
PROCESSED    = ROOT / "data" / "processed"
DATASET_OUT  = PROCESSED / "phishtrace_dataset.csv"
PROCESSED.mkdir(parents=True, exist_ok=True)

# ── Source definitions ────────────────────────────────────────────────────────
PHISHTANK_URL = "https://data.phishtank.com/data/online-valid.json.gz"
TRANCO_URL    = "https://tranco-list.eu/top-1m.csv.zip"

# ── Curated legitimate URLs ───────────────────────────────────────────────────
# Covers patterns the model misclassifies: numeric IDs in paths, dates in URLs,
# long paths, many slashes — all features over-associated with phishing.
EXTRA_LEGIT_URLS = [
    # ── Stack Overflow (numeric question IDs) ─────────────────────────────────
    "https://stackoverflow.com/questions/11227809/why-is-processing-a-sorted-array-faster",
    "https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do",
    "https://stackoverflow.com/questions/3294889/iterating-over-dictionaries-using-for-loops",
    "https://stackoverflow.com/questions/1732348/regex-match-open-tags-except-xhtml-self-contained",
    "https://stackoverflow.com/questions/36932/how-can-i-represent-an-enum-in-python",
    "https://stackoverflow.com/questions/2612096/what-does-if-name-main-do",
    "https://stackoverflow.com/questions/17301871/convert-string-to-int-in-python",
    "https://stackoverflow.com/questions/4843158/check-if-a-python-list-item-contains-a-string",
    "https://stackoverflow.com/questions/1436703/difference-between-str-and-repr",
    "https://stackoverflow.com/questions/6470428/catch-multiple-exceptions-in-one-line",
    "https://stackoverflow.com/questions/9371238/why-is-reading-lines-from-stdin-much-slower-in-c",
    "https://stackoverflow.com/questions/8381193/handle-large-file-uploads-with-django",
    "https://stackoverflow.com/questions/14797234/python-unittest-mock-patch",
    "https://stackoverflow.com/questions/22150635/how-to-use-virtualenv-with-python",
    "https://stackoverflow.com/questions/5137497/find-current-directory-and-files-directory",
    "https://stackoverflow.com/questions/4906977/how-to-access-environment-variable-values",
    "https://stackoverflow.com/questions/273192/how-do-i-create-a-directory-if-it-does-not-exist",
    "https://stackoverflow.com/questions/3207219/how-do-i-list-all-files-of-a-directory",
    "https://stackoverflow.com/questions/7588511/format-a-datetime-into-a-string-with-milliseconds",
    "https://stackoverflow.com/questions/6088077/how-to-get-a-random-element-from-a-list",
    "https://stackoverflow.com/questions/9536714/python-sort-dictionary-by-value",
    "https://stackoverflow.com/questions/72899/how-do-i-sort-a-list-of-dictionaries-by-a-value",
    "https://stackoverflow.com/questions/1602934/check-if-a-given-key-already-exists-in-a-dictionary",
    "https://stackoverflow.com/questions/3679294/how-to-check-if-a-string-contains-a-substring",
    "https://stackoverflow.com/questions/5618878/how-to-convert-list-to-string",
    "https://stackoverflow.com/questions/6797984/how-to-convert-string-to-lowercase-in-python",
    "https://stackoverflow.com/questions/4071796/how-to-get-the-number-of-elements-in-a-list",
    "https://stackoverflow.com/questions/8369219/how-to-read-a-text-file-into-a-string-variable",
    "https://stackoverflow.com/questions/11416324/python-pandas-read-csv",
    "https://stackoverflow.com/questions/18837262/merge-multiple-dataframes-pandas",
    "https://stackoverflow.com/questions/1051819/what-is-the-difference-between-append-and-extend",
    "https://stackoverflow.com/questions/5214578/print-string-to-text-file",
    "https://stackoverflow.com/questions/13784263/regex-email-validation",
    "https://stackoverflow.com/questions/2257441/random-string-generation-with-upper-case-letters",
    "https://stackoverflow.com/questions/714881/list-vs-tuple-when-to-use-each",
    "https://stackoverflow.com/questions/4984647/accessing-dict-keys-like-an-attribute",
    "https://stackoverflow.com/questions/12897374/get-unique-values-from-a-list-in-python",
    "https://stackoverflow.com/questions/7837722/what-is-the-most-efficient-way-to-loop-through-dataframes",
    "https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-in-a-dataframe-in-pandas",
    "https://stackoverflow.com/questions/29765620/pandas-merge-two-dataframes",
    "https://stackoverflow.com/questions/23668427/pandas-three-dataframes-merge",
    "https://stackoverflow.com/questions/25692915/pandas-groupby-count-distinct-values",
    "https://stackoverflow.com/questions/17071871/select-rows-from-a-dataframe-based-on-values",
    "https://stackoverflow.com/questions/21800169/python-pandas-get-index-of-rows-which-column",
    "https://stackoverflow.com/questions/10715965/add-one-row-to-pandas-dataframe",
    "https://stackoverflow.com/questions/19828822/how-to-check-whether-a-pandas-dataframe-is-empty",
    "https://stackoverflow.com/questions/12703822/python-flask-how-to-remove-cache",
    "https://stackoverflow.com/questions/30362391/how-do-you-find-the-first-key-in-a-dictionary",
    "https://stackoverflow.com/questions/53513/how-do-i-check-if-a-list-is-empty",
    "https://stackoverflow.com/questions/2474015/getting-the-index-of-the-returned-max-or-min-item",

    # ── GitHub (issues, PRs, commits) ─────────────────────────────────────────
    "https://github.com/python/cpython/issues/91234",
    "https://github.com/python/cpython/issues/88123",
    "https://github.com/django/django/issues/15678",
    "https://github.com/django/django/pull/16234",
    "https://github.com/pallets/flask/issues/4921",
    "https://github.com/pallets/flask/pull/4856",
    "https://github.com/numpy/numpy/issues/21936",
    "https://github.com/pandas-dev/pandas/issues/47631",
    "https://github.com/scikit-learn/scikit-learn/issues/23457",
    "https://github.com/torvalds/linux/commit/a2b4c6d8e0f1",
    "https://github.com/microsoft/vscode/issues/156789",
    "https://github.com/microsoft/vscode/pull/154321",
    "https://github.com/facebook/react/issues/24521",
    "https://github.com/facebook/react/pull/24398",
    "https://github.com/vercel/next.js/issues/41234",
    "https://github.com/vercel/next.js/pull/40987",
    "https://github.com/vuejs/vue/issues/12456",
    "https://github.com/angular/angular/issues/47123",
    "https://github.com/rust-lang/rust/issues/98765",
    "https://github.com/golang/go/issues/54321",
    "https://github.com/nodejs/node/issues/45678",
    "https://github.com/kubernetes/kubernetes/issues/112345",
    "https://github.com/docker/compose/issues/9876",
    "https://github.com/tensorflow/tensorflow/issues/56789",
    "https://github.com/pytorch/pytorch/issues/78901",
    "https://github.com/huggingface/transformers/issues/19876",
    "https://github.com/axios/axios/issues/5123",
    "https://github.com/expressjs/express/issues/4987",
    "https://github.com/sveltejs/svelte/issues/7654",
    "https://github.com/tailwindlabs/tailwindcss/issues/9123",
    "https://github.com/vitejs/vite/issues/10234",
    "https://github.com/denoland/deno/issues/16543",
    "https://github.com/rust-lang/cargo/issues/11234",
    "https://github.com/mozilla/firefox/issues/1234567",
    "https://github.com/electron/electron/issues/35678",
    "https://github.com/redis/redis/issues/10987",
    "https://github.com/postgres/postgres/issues/456",
    "https://github.com/hashicorp/terraform/issues/31456",
    "https://github.com/ansible/ansible/issues/78234",
    "https://github.com/grafana/grafana/issues/56123",

    # ── Wikipedia (articles with long paths) ──────────────────────────────────
    "https://en.wikipedia.org/wiki/Python_(programming_language)",
    "https://en.wikipedia.org/wiki/Machine_learning",
    "https://en.wikipedia.org/wiki/Artificial_intelligence",
    "https://en.wikipedia.org/wiki/World_Wide_Web",
    "https://en.wikipedia.org/wiki/Computer_science",
    "https://en.wikipedia.org/wiki/Cryptography",
    "https://en.wikipedia.org/wiki/Transport_Layer_Security",
    "https://en.wikipedia.org/wiki/Phishing",
    "https://en.wikipedia.org/wiki/Cybersecurity",
    "https://en.wikipedia.org/wiki/Deep_learning",
    "https://en.wikipedia.org/wiki/Natural_language_processing",
    "https://en.wikipedia.org/wiki/Convolutional_neural_network",
    "https://en.wikipedia.org/wiki/Random_forest",
    "https://en.wikipedia.org/wiki/Support_vector_machine",
    "https://en.wikipedia.org/wiki/K-means_clustering",
    "https://en.wikipedia.org/wiki/Principal_component_analysis",
    "https://en.wikipedia.org/wiki/Gradient_boosting",
    "https://en.wikipedia.org/wiki/Recurrent_neural_network",
    "https://en.wikipedia.org/wiki/Transformer_(machine_learning_model)",
    "https://en.wikipedia.org/wiki/BERT_(language_model)",
    "https://en.wikipedia.org/wiki/Long_short-term_memory",
    "https://en.wikipedia.org/wiki/Generative_adversarial_network",
    "https://en.wikipedia.org/wiki/Internet_protocol_suite",
    "https://en.wikipedia.org/wiki/Domain_Name_System",
    "https://en.wikipedia.org/wiki/Hypertext_Transfer_Protocol",
    "https://en.wikipedia.org/wiki/Unix_philosophy",
    "https://en.wikipedia.org/wiki/Open_source",
    "https://en.wikipedia.org/wiki/Linux_kernel",
    "https://en.wikipedia.org/wiki/Git_(software)",
    "https://en.wikipedia.org/wiki/Docker_(software)",

    # ── YouTube (video IDs — alphanumeric, 11 chars) ───────────────────────────
    "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
    "https://www.youtube.com/watch?v=rfscVS0vtbw",
    "https://www.youtube.com/watch?v=_uQrJ0TkZlc",
    "https://www.youtube.com/watch?v=ZyhVh-qRZPA",
    "https://www.youtube.com/watch?v=kqtD5dpn9C8",
    "https://www.youtube.com/watch?v=8jLOx1hD3_o",
    "https://www.youtube.com/watch?v=HXV3zeQKqGY",
    "https://www.youtube.com/watch?v=tKTZoB2Vjuk",
    "https://www.youtube.com/watch?v=5MgBikgcWnY",
    "https://www.youtube.com/watch?v=ysz5S6PUM-U",
    "https://www.youtube.com/watch?v=7DKv5H5Frt0",
    "https://www.youtube.com/watch?v=0jgrCKhxE1s",
    "https://www.youtube.com/watch?v=fis26HvvDII",
    "https://www.youtube.com/watch?v=x7X9w_GIm1s",
    "https://www.youtube.com/watch?v=Wjrpf6fBTqY",
    "https://www.youtube.com/watch?v=qiH4UbxnVyY",
    "https://www.youtube.com/watch?v=aircAruvnKk",
    "https://www.youtube.com/watch?v=IHZwWFHWa-w",
    "https://www.youtube.com/watch?v=Ilg3gGewQ5U",
    "https://www.youtube.com/watch?v=OkmNXy7er84",

    # ── Reddit (posts with alphanumeric IDs) ──────────────────────────────────
    "https://www.reddit.com/r/Python/comments/10abc12/python_tips_for_beginners",
    "https://www.reddit.com/r/MachineLearning/comments/11xyz34/new_paper_on_transformers",
    "https://www.reddit.com/r/programming/comments/9qwe56/why_functional_programming_matters",
    "https://www.reddit.com/r/learnpython/comments/8rty78/help_with_pandas_dataframe",
    "https://www.reddit.com/r/webdev/comments/7uvw90/best_practices_for_rest_apis",
    "https://www.reddit.com/r/netsec/comments/6abc11/new_phishing_technique_discovered",
    "https://www.reddit.com/r/cybersecurity/comments/5def22/how_to_detect_phishing_urls",
    "https://www.reddit.com/r/datascience/comments/4ghi33/random_forest_vs_xgboost",
    "https://www.reddit.com/r/javascript/comments/3jkl44/react_18_new_features",
    "https://www.reddit.com/r/linux/comments/2mno55/kernel_6_0_released",
    "https://www.reddit.com/r/Python/comments/1pqr66/best_python_libraries_2024",
    "https://www.reddit.com/r/cscareerquestions/comments/stu77/switching_from_backend_to_ml",
    "https://www.reddit.com/r/golang/comments/vwx88/go_generics_tutorial",
    "https://www.reddit.com/r/rust/comments/yza99/rust_for_beginners_roadmap",
    "https://www.reddit.com/r/devops/comments/bcd00/kubernetes_vs_docker_swarm",

    # ── News articles (dates in paths, numeric IDs) ────────────────────────────
    "https://www.bbc.com/news/technology-58927890",
    "https://www.bbc.com/news/world-us-canada-59876543",
    "https://www.bbc.com/news/business-57654321",
    "https://www.bbc.com/news/science-environment-56789012",
    "https://www.bbc.com/news/health-55432109",
    "https://edition.cnn.com/2024/01/15/tech/cybersecurity-report/index.html",
    "https://edition.cnn.com/2024/03/22/business/ai-regulations/index.html",
    "https://edition.cnn.com/2023/11/08/tech/python-programming/index.html",
    "https://www.reuters.com/technology/cybersecurity/phishing-attacks-rise-2024-01-15",
    "https://www.reuters.com/business/finance/banking-cybercrime-losses-2023-11-20",
    "https://www.theguardian.com/technology/2024/jan/15/ai-cybersecurity-threats",
    "https://www.theguardian.com/technology/2023/dec/01/python-programming-language",
    "https://techcrunch.com/2024/01/10/machine-learning-security-2024",
    "https://techcrunch.com/2023/12/15/open-source-security-vulnerabilities",
    "https://arstechnica.com/security/2024/01/15/phishing-detection-ai",
    "https://arstechnica.com/gadgets/2024/02/10/linux-kernel-6-7-released",
    "https://www.wired.com/story/ai-phishing-detection-2024",
    "https://www.wired.com/story/best-password-managers-reviewed-2024",
    "https://www.theverge.com/2024/1/15/24038901/python-most-popular-language",
    "https://www.theverge.com/2023/12/5/23988901/open-source-ai-tools",
    "https://nytimes.com/2024/01/14/technology/cybersecurity-threats.html",
    "https://www.washingtonpost.com/technology/2024/01/10/ai-security-risks",
    "https://apnews.com/article/cybersecurity-phishing-2024-a1b2c3d4e5f6",
    "https://www.zdnet.com/article/best-cybersecurity-tools-2024",
    "https://www.csoonline.com/article/3715234/phishing-prevention-guide.html",

    # ── Developer documentation ────────────────────────────────────────────────
    "https://docs.python.org/3/library/os.path.html",
    "https://docs.python.org/3/library/collections.html",
    "https://docs.python.org/3/library/itertools.html",
    "https://docs.python.org/3/library/functools.html",
    "https://docs.python.org/3/library/pathlib.html",
    "https://docs.python.org/3/library/json.html",
    "https://docs.python.org/3/library/re.html",
    "https://docs.python.org/3/library/datetime.html",
    "https://docs.python.org/3/library/logging.html",
    "https://docs.python.org/3/library/unittest.html",
    "https://docs.python.org/3/tutorial/classes.html",
    "https://docs.python.org/3/tutorial/datastructures.html",
    "https://docs.djangoproject.com/en/4.2/ref/models/querysets",
    "https://docs.djangoproject.com/en/4.2/topics/db/queries",
    "https://docs.djangoproject.com/en/4.2/ref/settings",
    "https://flask.palletsprojects.com/en/3.0.x/quickstart",
    "https://flask.palletsprojects.com/en/3.0.x/api",
    "https://fastapi.tiangolo.com/tutorial/path-params",
    "https://fastapi.tiangolo.com/tutorial/query-params",
    "https://fastapi.tiangolo.com/tutorial/body",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Reference/Global_Objects/Promise",
    "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API/Using_Fetch",
    "https://developer.mozilla.org/en-US/docs/Web/JavaScript/Guide/Regular_Expressions",
    "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_Grid_Layout",
    "https://developer.mozilla.org/en-US/docs/Web/HTML/Element/input",
    "https://react.dev/learn/passing-props-to-a-component",
    "https://react.dev/learn/managing-state",
    "https://nextjs.org/docs/app/building-your-application/routing",
    "https://nextjs.org/docs/app/building-your-application/data-fetching",
    "https://vuejs.org/guide/essentials/reactivity-fundamentals",

    # ── Microsoft / Azure / Windows docs ─────────────────────────────────────
    "https://learn.microsoft.com/en-us/azure/security/fundamentals/overview",
    "https://learn.microsoft.com/en-us/dotnet/csharp/programming-guide",
    "https://learn.microsoft.com/en-us/windows/win32/api/winbase",
    "https://support.microsoft.com/en-us/windows/update-windows-3c5ae7fc-9fb6-9af1",
    "https://support.microsoft.com/en-us/office/excel-functions-by-category-5f91f4e9",
    "https://learn.microsoft.com/en-us/azure/active-directory/fundamentals/whatis",
    "https://learn.microsoft.com/en-us/power-bi/fundamentals/service-get-started",
    "https://docs.microsoft.com/en-us/sql/t-sql/statements/select-transact-sql",
    "https://learn.microsoft.com/en-us/visualstudio/get-started/csharp/tutorial-console",
    "https://learn.microsoft.com/en-us/azure/machine-learning/overview-what-is-azure-ml",

    # ── Apple support ────────────────────────────────────────────────────────
    "https://support.apple.com/en-us/HT201229",
    "https://support.apple.com/en-us/HT204085",
    "https://support.apple.com/en-us/HT212927",
    "https://support.apple.com/en-us/HT213078",
    "https://support.apple.com/en-us/HT204611",
    "https://developer.apple.com/documentation/swift/array",
    "https://developer.apple.com/documentation/foundation/url",
    "https://developer.apple.com/documentation/uikit/uiviewcontroller",
    "https://developer.apple.com/documentation/swiftui/text",
    "https://developer.apple.com/documentation/security/certificate_key_and_trust_services",

    # ── Google developers / support ───────────────────────────────────────────
    "https://developers.google.com/safe-browsing/v4/lookup-api",
    "https://developers.google.com/maps/documentation/javascript/overview",
    "https://cloud.google.com/security/products/threat-intelligence",
    "https://support.google.com/accounts/answer/185839",
    "https://support.google.com/chrome/answer/95647",
    "https://cloud.google.com/bigquery/docs/introduction",
    "https://firebase.google.com/docs/web/setup",
    "https://developers.google.com/analytics/devguides/reporting/core/v4",
    "https://cloud.google.com/storage/docs/creating-buckets",
    "https://developers.google.com/gmail/api/reference/rest",

    # ── Amazon AWS docs ───────────────────────────────────────────────────────
    "https://docs.aws.amazon.com/s3/latest/userguide/getting-started.html",
    "https://docs.aws.amazon.com/lambda/latest/dg/welcome.html",
    "https://docs.aws.amazon.com/ec2/latest/userguide/concepts.html",
    "https://docs.aws.amazon.com/iam/latest/userguide/introduction.html",
    "https://docs.aws.amazon.com/vpc/latest/userguide/what-is-amazon-vpc.html",
    "https://docs.aws.amazon.com/cloudwatch/latest/logs/WhatIsCloudWatchLogs.html",
    "https://docs.aws.amazon.com/rds/latest/userguide/Welcome.html",
    "https://docs.aws.amazon.com/sagemaker/latest/dg/whatis.html",
    "https://repost.aws/knowledge-center/s3-presigned-url",
    "https://repost.aws/knowledge-center/lambda-function-timeout",

    # ── Package registries (PyPI, npm) ────────────────────────────────────────
    "https://pypi.org/project/requests/2.31.0",
    "https://pypi.org/project/numpy/1.26.4",
    "https://pypi.org/project/pandas/2.2.0",
    "https://pypi.org/project/scikit-learn/1.4.0",
    "https://pypi.org/project/flask/3.0.2",
    "https://pypi.org/project/django/5.0.2",
    "https://pypi.org/project/fastapi/0.109.2",
    "https://pypi.org/project/tensorflow/2.15.0",
    "https://pypi.org/project/pytorch/2.2.0",
    "https://pypi.org/project/xgboost/2.0.3",
    "https://www.npmjs.com/package/react/v/18.2.0",
    "https://www.npmjs.com/package/typescript/v/5.3.3",
    "https://www.npmjs.com/package/next/v/14.1.0",
    "https://www.npmjs.com/package/express/v/4.18.2",
    "https://www.npmjs.com/package/axios/v/1.6.7",
    "https://www.npmjs.com/package/lodash/v/4.17.21",
    "https://www.npmjs.com/package/tailwindcss/v/3.4.1",
    "https://www.npmjs.com/package/vite/v/5.1.1",
    "https://www.npmjs.com/package/eslint/v/8.57.0",
    "https://www.npmjs.com/package/jest/v/29.7.0",

    # ── Government / .gov sites ───────────────────────────────────────────────
    "https://www.cisa.gov/topics/cyber-threats-and-advisories/phishing",
    "https://www.cisa.gov/resources-tools/resources/free-cybersecurity-services-and-tools",
    "https://www.ftc.gov/business-guidance/small-businesses/cybersecurity/phishing",
    "https://www.ftc.gov/news-events/news/press-releases/2024/01/ftc-releases-report",
    "https://www.nist.gov/cybersecurity/framework",
    "https://nvd.nist.gov/vuln/detail/CVE-2024-12345",
    "https://nvd.nist.gov/vuln/detail/CVE-2023-44487",
    "https://www.ncsc.gov.uk/guidance/phishing",
    "https://www.justice.gov/criminal-fraud/internet-fraud",
    "https://www.ic3.gov/Home/AnnualReports",

    # ── Academic / .edu ───────────────────────────────────────────────────────
    "https://web.mit.edu/6.005/www/fa15/classes/10-abstract-data-types",
    "https://cs231n.stanford.edu/2024/syllabus.html",
    "https://www.cs.cornell.edu/courses/cs4780/2024sp",
    "https://ocw.mit.edu/courses/6-0001-introduction-to-computer-science-and-programming",
    "https://www.coursera.org/learn/machine-learning/lecture/db3jS/what-is-machine-learning",
    "https://arxiv.org/abs/2301.07094",
    "https://arxiv.org/abs/1706.03762",
    "https://arxiv.org/abs/2005.14165",
    "https://arxiv.org/abs/1810.04805",
    "https://arxiv.org/abs/2303.08774",

    # ── Hacker News ───────────────────────────────────────────────────────────
    "https://news.ycombinator.com/item?id=38987654",
    "https://news.ycombinator.com/item?id=37654321",
    "https://news.ycombinator.com/item?id=36543210",
    "https://news.ycombinator.com/item?id=35432109",
    "https://news.ycombinator.com/item?id=34321098",
    "https://news.ycombinator.com/item?id=33210987",
    "https://news.ycombinator.com/item?id=32109876",
    "https://news.ycombinator.com/item?id=31098765",
    "https://news.ycombinator.com/item?id=39876543",
    "https://news.ycombinator.com/item?id=40123456",

    # ── Archive.org / Wayback Machine ─────────────────────────────────────────
    "https://web.archive.org/web/20240101000000/https://www.python.org",
    "https://web.archive.org/web/20231215120000/https://www.github.com",
    "https://web.archive.org/web/20230601000000/https://stackoverflow.com",
    "https://archive.org/details/python_tutorial_2024",
    "https://archive.org/details/machine_learning_course",

    # ── Docker Hub / container registries ────────────────────────────────────
    "https://hub.docker.com/r/python/tags",
    "https://hub.docker.com/r/nginx/tags",
    "https://hub.docker.com/r/postgres/tags",
    "https://hub.docker.com/r/redis/tags",
    "https://hub.docker.com/layers/python/library/python/3.11-slim/images/sha256-abc123",

    # ── OWASP / security resources ────────────────────────────────────────────
    "https://owasp.org/www-project-top-ten",
    "https://owasp.org/www-community/attacks/Phishing",
    "https://owasp.org/www-project-web-security-testing-guide",
    "https://cheatsheetseries.owasp.org/cheatsheets/Phishing_Prevention_Cheat_Sheet.html",
    "https://cheatsheetseries.owasp.org/cheatsheets/Authentication_Cheat_Sheet.html",

    # ── Stack Exchange network ────────────────────────────────────────────────
    "https://security.stackexchange.com/questions/249190/how-to-detect-phishing",
    "https://security.stackexchange.com/questions/238765/url-analysis-machine-learning",
    "https://datascience.stackexchange.com/questions/98765/random-forest-feature-importance",
    "https://unix.stackexchange.com/questions/148543/how-to-set-environment-variables",
    "https://serverfault.com/questions/145777/nginx-reverse-proxy-configuration",
    "https://superuser.com/questions/138587/how-to-use-wget-to-download-files",
    "https://askubuntu.com/questions/1234567/how-to-install-python-3-11",
    "https://dba.stackexchange.com/questions/234567/postgresql-index-optimization",
    "https://networkengineering.stackexchange.com/questions/12345/dns-resolution-process",
    "https://crypto.stackexchange.com/questions/56789/tls-handshake-explained",

    # ── LinkedIn / professional ───────────────────────────────────────────────
    "https://www.linkedin.com/learning/python-essential-training-14898805",
    "https://www.linkedin.com/learning/machine-learning-foundations-statistics",
    "https://www.linkedin.com/pulse/detecting-phishing-urls-machine-learning-article-12345678",

    # ── Kaggle datasets / notebooks ───────────────────────────────────────────
    "https://www.kaggle.com/datasets/eswarchandt/phishing-website-detector",
    "https://www.kaggle.com/code/karnikakapoor/phishing-url-detection-ml",
    "https://www.kaggle.com/datasets/sid321axn/malicious-urls-dataset",
    "https://www.kaggle.com/code/shashwatwork/phishing-website-detection-rf",
    "https://www.kaggle.com/competitions/web-traffic-time-series-forecasting",

    # ── Misc tech blogs / tutorials with numeric IDs ─────────────────────────
    "https://towardsdatascience.com/phishing-url-detection-with-ml-5b4e6df2b37a",
    "https://medium.com/better-programming/python-url-feature-extraction-12345678",
    "https://dev.to/username/building-url-classifier-python-2024-abc123",
    "https://hashnode.dev/how-to-build-phishing-detector-12345678",
    "https://www.freecodecamp.org/news/machine-learning-phishing-detection-12345678",
    "https://realpython.com/python-web-scraping-practical-introduction",
    "https://realpython.com/python-f-strings",
    "https://realpython.com/python-concurrency",
    "https://realpython.com/async-io-python",
    "https://www.digitalocean.com/community/tutorials/how-to-use-flask-with-nginx",
    "https://www.digitalocean.com/community/tutorials/how-to-install-python-3-ubuntu-20-04",

    # ── E-commerce (Amazon product pages) ────────────────────────────────────
    "https://www.amazon.com/dp/B08N5WRWNW/ref=sr_1_1",
    "https://www.amazon.com/dp/B07FZ8S74R/ref=sr_1_2",
    "https://www.amazon.com/s?k=python+programming&ref=nb_sb_noss",
    "https://www.amazon.com/Python-Crash-Course-Eric-Matthes/dp/1593279280",
    "https://www.bestbuy.com/site/laptop/14987654.p",
    "https://www.newegg.com/p/N82E16824001234",

    # ── Misc legitimate numeric-heavy URLs ───────────────────────────────────
    "https://www.imdb.com/title/tt0111161",
    "https://www.imdb.com/title/tt0068646",
    "https://www.imdb.com/title/tt0468569",
    "https://letterboxd.com/film/the-godfather",
    "https://www.goodreads.com/book/show/5907.The_Hitchhiker_s_Guide_to_the_Galaxy",
    "https://www.goodreads.com/book/show/11588.Ender_s_Game",
    "https://open.spotify.com/track/4uLU6hMCjMI75M1A2tKUQC",
    "https://open.spotify.com/album/4eLPsYPBmXABThSJ821sqY",
    "https://soundcloud.com/user-12345678/track-title-here",
    "https://bandcamp.com/track/12345678",
    "https://www.eventbrite.com/e/python-workshop-2024-tickets-123456789",
    "https://meetup.com/python-developers/events/987654321",
    "https://trello.com/b/abcdefgh/project-board",
    "https://jira.atlassian.com/browse/JIRA-12345",
    "https://confluence.atlassian.com/doc/page/123456789",
    "https://www.wolframalpha.com/input?i=prime+factorization+of+12345678",
    "https://regexr.com/3cr6f",
    "https://regex101.com/r/abc123/1",
    "https://jsonplaceholder.typicode.com/posts/12345",
    "https://httpbin.org/status/200",

    # ── More Stack Overflow ───────────────────────────────────────────────────
    "https://stackoverflow.com/questions/10048571/python-finding-a-trend-in-a-set-of-numbers",
    "https://stackoverflow.com/questions/11664443/python-list-comprehension-vs-map",
    "https://stackoverflow.com/questions/12435169/numpy-array-indexing",
    "https://stackoverflow.com/questions/13411544/delete-a-column-from-a-pandas-dataframe",
    "https://stackoverflow.com/questions/14529523/python-split-string-by-multiple-delimiters",
    "https://stackoverflow.com/questions/15374557/how-do-i-copy-a-file-in-python",
    "https://stackoverflow.com/questions/16476924/how-to-iterate-over-rows-pandas",
    "https://stackoverflow.com/questions/17778394/list-highest-occurring-elements-in-a-list",
    "https://stackoverflow.com/questions/18688083/python-get-all-possible-subsequences",
    "https://stackoverflow.com/questions/19482970/get-list-from-pandas-dataframe-column",
    "https://stackoverflow.com/questions/20546333/python-read-json-file",
    "https://stackoverflow.com/questions/21563769/how-to-remove-empty-strings-from-a-list",
    "https://stackoverflow.com/questions/22508590/encode-decode-strings-in-python-3",
    "https://stackoverflow.com/questions/23586510/return-value-from-one-function-to-another",
    "https://stackoverflow.com/questions/24678308/how-to-find-percentile-stats-of-given-list",
    "https://stackoverflow.com/questions/25561031/replace-all-occurrences-of-string-in-list",
    "https://stackoverflow.com/questions/26646362/numpy-array-is-not-json-serializable",
    "https://stackoverflow.com/questions/27779093/get-list-of-all-classes-within-current-module",
    "https://stackoverflow.com/questions/28789452/python-reverse-dictionary-lookup",
    "https://stackoverflow.com/questions/29751557/exclude-a-character-set-in-python-regex",

    # ── More GitHub ───────────────────────────────────────────────────────────
    "https://github.com/openai/openai-python/issues/890",
    "https://github.com/openai/whisper/issues/1234",
    "https://github.com/ollama/ollama/issues/2345",
    "https://github.com/langchain-ai/langchain/issues/15678",
    "https://github.com/fastapi/fastapi/issues/10123",
    "https://github.com/tiangolo/sqlmodel/issues/567",
    "https://github.com/encode/httpx/issues/2456",
    "https://github.com/pydantic/pydantic/issues/8901",
    "https://github.com/astral-sh/ruff/issues/5678",
    "https://github.com/python-poetry/poetry/issues/7890",
    "https://github.com/pypa/pip/issues/12345",
    "https://github.com/conda/conda/issues/13456",
    "https://github.com/jupyter/notebook/issues/6789",
    "https://github.com/matplotlib/matplotlib/issues/25678",
    "https://github.com/bokeh/bokeh/issues/13234",
    "https://github.com/plotly/plotly.py/issues/4321",
    "https://github.com/streamlit/streamlit/issues/7654",
    "https://github.com/gradio-app/gradio/issues/6543",
    "https://github.com/celery/celery/issues/8765",
    "https://github.com/sqlalchemy/sqlalchemy/issues/10987",

    # ── More news articles with dates in path ─────────────────────────────────
    "https://www.bbc.com/news/technology/2024/01/15/ai-phishing-detection",
    "https://techcrunch.com/2024/02/20/startup-raises-50-million-cybersecurity",
    "https://arstechnica.com/security/2024/03/10/new-phishing-campaign-targets-banks",
    "https://www.reuters.com/technology/2024/01/20/google-updates-safe-browsing-api",
    "https://www.theguardian.com/technology/2024/feb/15/machine-learning-security",
    "https://www.wired.com/story/phishing-ai-detection-2024-overview",
    "https://www.csoonline.com/article/3812345/top-phishing-trends-2024.html",
    "https://www.darkreading.com/threat-intelligence/phishing-attacks-increased-2024",
    "https://www.bleepingcomputer.com/news/security/new-phishing-technique-bypass-mfa",
    "https://krebsonsecurity.com/2024/01/phishing-kit-analysis-2024",
    "https://threatpost.com/phishing-detection-machine-learning/183456",
    "https://www.infosecurity-magazine.com/news/phishing-urls-ml-detection-2024",
    "https://www.securityweek.com/phishing-url-analysis-machine-learning-approaches",
    "https://www.helpnetsecurity.com/2024/01/15/phishing-detection-tools",

    # ── More docs with long numeric/structured paths ───────────────────────────
    "https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
    "https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html",
    "https://scikit-learn.org/stable/modules/model_evaluation.html",
    "https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html",
    "https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.groupby.html",
    "https://pandas.pydata.org/docs/reference/api/pandas.read_csv.html",
    "https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html",
    "https://seaborn.pydata.org/generated/seaborn.heatmap.html",
    "https://xgboost.readthedocs.io/en/stable/python/python_api.html",
    "https://shap.readthedocs.io/en/latest/example_notebooks/overviews/An_introduction_to_explainable_AI_with_Shapley_values.html",
    "https://docs.sqlalchemy.org/en/20/orm/quickstart.html",
    "https://celery-5-0-0.readthedocs.io/en/stable/getting-started/introduction.html",
    "https://redis-py.readthedocs.io/en/stable/commands.html",
    "https://boto3.amazonaws.com/v1/documentation/api/latest/reference/services/s3.html",
    "https://kubernetes.io/docs/concepts/workloads/controllers/deployment",
    "https://docs.docker.com/compose/compose-file/compose-file-v3",
    "https://helm.sh/docs/intro/using_helm",
    "https://www.postgresql.org/docs/16/sql-select.html",
    "https://dev.mysql.com/doc/refman/8.0/en/select.html",
    "https://www.mongodb.com/docs/manual/reference/operator/aggregation/group",

    # ── More e-commerce and product pages ─────────────────────────────────────
    "https://www.amazon.com/Fluent-Python-Concise-Effective-Programming/dp/1492056359",
    "https://www.amazon.com/Clean-Code-Handbook-Software-Craftsmanship/dp/0132350882",
    "https://www.amazon.com/Design-Patterns-Elements-Reusable-Object-Oriented/dp/0201633612",
    "https://www.ebay.com/itm/laptop-dell-xps-15/234567890123",
    "https://www.ebay.com/itm/mechanical-keyboard-das/345678901234",
    "https://www.etsy.com/listing/987654321/handmade-python-programming-mug",
    "https://www.walmart.com/ip/python-books/234567890",
    "https://www.target.com/p/laptop-stand/-/A-12345678",
    "https://www.bestbuy.com/site/apple-macbook-pro/6509652.p",
    "https://www.newegg.com/p/1B4-00Y5-00007",

    # ── More academic / research papers ───────────────────────────────────────
    "https://arxiv.org/abs/2010.11929",
    "https://arxiv.org/abs/2104.09864",
    "https://arxiv.org/abs/2106.09685",
    "https://arxiv.org/abs/2111.01998",
    "https://arxiv.org/abs/2204.02311",
    "https://arxiv.org/abs/2302.13971",
    "https://arxiv.org/abs/2307.09288",
    "https://arxiv.org/abs/2401.12345",
    "https://papers.nips.cc/paper_files/paper/2017/hash/3f5ee243547dee91fbd053c1c4a845aa-Abstract.html",
    "https://proceedings.mlr.press/v139/radford21a.html",
    "https://dl.acm.org/doi/10.1145/3447548.3467385",
    "https://ieeexplore.ieee.org/document/9123456",
    "https://www.sciencedirect.com/science/article/pii/S0167404823001234",
    "https://link.springer.com/article/10.1007/s10994-023-06345-7",
    "https://scholar.google.com/scholar?q=phishing+url+detection+machine+learning",

    # ── More Stack Overflow (final batch) ─────────────────────────────────────
    "https://stackoverflow.com/questions/30650474/python-rename-specific-column",
    "https://stackoverflow.com/questions/31645466/give-column-name-when-read-csv-file-pandas",
    "https://stackoverflow.com/questions/32400867/pandas-read-csv-from-url",
    "https://stackoverflow.com/questions/33190313/pandas-plot-bar-chart-grouped",
    "https://stackoverflow.com/questions/34226400/find-the-index-of-a-value-in-a-list",
    "https://stackoverflow.com/questions/35276599/how-do-i-check-if-directory-exists-python",
    "https://stackoverflow.com/questions/36428323/python-list-to-json",
    "https://stackoverflow.com/questions/37427888/write-list-to-csv-python",
    "https://stackoverflow.com/questions/38477288/python-parse-html-beautifulsoup",
    "https://stackoverflow.com/questions/39590234/how-do-you-flatten-a-list-of-lists",
    "https://stackoverflow.com/questions/40696492/python-apply-function-to-list",
    "https://stackoverflow.com/questions/41796890/python-class-inheritance-example",
    "https://stackoverflow.com/questions/42867900/python-dictionary-comprehension",
    "https://stackoverflow.com/questions/43908091/python-walrus-operator-use-cases",
    "https://stackoverflow.com/questions/44939011/python-dataclass-vs-namedtuple",
    "https://stackoverflow.com/questions/45970134/python-type-hints-best-practices",
    "https://stackoverflow.com/questions/46001123/python-abstract-base-class-example",
    "https://stackoverflow.com/questions/47032012/python-context-manager-with-statement",
    "https://stackoverflow.com/questions/48063013/python-generator-yield-from",
    "https://stackoverflow.com/questions/49094190/python-async-await-example",
    "https://stackoverflow.com/questions/50125176/python-multiprocessing-pool-map",
    "https://stackoverflow.com/questions/51156253/python-threading-lock-example",
    "https://stackoverflow.com/questions/52187311/difference-between-is-and-equals",
    "https://stackoverflow.com/questions/53198312/python-property-decorator-example",
    "https://stackoverflow.com/questions/54209213/python-slots-memory-usage",
    "https://stackoverflow.com/questions/55220314/python-weakref-explanation",
    "https://stackoverflow.com/questions/56231315/python-descriptor-protocol",
    "https://stackoverflow.com/questions/57242416/python-metaclass-example",
    "https://stackoverflow.com/questions/58253517/python-dunder-methods-list",
    "https://stackoverflow.com/questions/59264618/python-operator-overloading",
    "https://stackoverflow.com/questions/60275719/python-functools-lru-cache",
    "https://stackoverflow.com/questions/61286720/python-itertools-combinations",
]


# ═══════════════════════════════════════════════════════════════════════════════
#  Loaders for each raw file
# ═══════════════════════════════════════════════════════════════════════════════
def load_phishing_csv(path: Path) -> pd.Series:
    """Load phishing URLs from a CSV. Filters by status=='phishing' when present."""
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "phishing"]
    col = next((c for c in df.columns if "url" in c.lower()), df.columns[0])
    return df[col].dropna().astype(str)


def load_openphish(path: Path) -> pd.Series:
    """One URL per line in openphish_feed.txt."""
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    return pd.Series([l.strip() for l in lines if l.strip().startswith("http")])


def load_tranco(path: Path) -> pd.Series:
    """Tranco CSV: rank,domain — convert domain → https://domain URL."""
    df = pd.read_csv(path)
    return df["domain"].dropna().apply(lambda d: f"https://{d}")


def load_verified_online(path: Path) -> pd.Series:
    """PhishTank verified_online CSV — column 'url' or first column."""
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")
    col = next((c for c in df.columns if "url" in c.lower()), df.columns[0])
    return df[col].dropna().astype(str)


# ═══════════════════════════════════════════════════════════════════════════════
#  Merge + deduplicate + label
# ═══════════════════════════════════════════════════════════════════════════════
def build_dataset() -> pd.DataFrame:
    """
    Merge all raw sources into one labelled DataFrame.
    label = 1 → phishing  |  label = 0 → legitimate
    """
    print("── Building dataset ──────────────────────────────────────────────────")
    phishing_urls  = []
    legit_urls     = []

    # Phishing sources
    for fname, loader in [
        ("dataset_phishing.csv",      load_phishing_csv),
        ("openphish_feed.txt",        load_openphish),
        ("verified_online.csv",        load_verified_online),
    ]:
        fpath = RAW_DIR / fname
        if fpath.exists():
            s = loader(fpath)
            phishing_urls.append(s)
            print(f"  [phishing] {fname:<30} → {len(s):>7,} URLs")
        else:
            print(f"  [skip]    {fname} not found")

    # Legitimate sources
    for fname, loader in [
        ("tranco_top10k.csv", load_tranco),
    ]:
        fpath = RAW_DIR / fname
        if fpath.exists():
            s = loader(fpath)
            legit_urls.append(s)
            print(f"  [legit]   {fname:<30} → {len(s):>7,} URLs")
        else:
            print(f"  [skip]    {fname} not found")

    # Curated supplement: diverse legitimate URLs with numeric paths
    extra = pd.Series(EXTRA_LEGIT_URLS).drop_duplicates()
    legit_urls.append(extra)
    print(f"  [legit]   EXTRA_LEGIT_URLS                → {len(extra):>7,} URLs")

    # Combine
    phish_series = pd.concat(phishing_urls).drop_duplicates().reset_index(drop=True)
    legit_series = pd.concat(legit_urls).drop_duplicates().reset_index(drop=True)

    phish_df = pd.DataFrame({"url": phish_series, "label": 1})
    legit_df = pd.DataFrame({"url": legit_series, "label": 0})

    # Balance: cap legit to 1.2× phishing count to avoid severe imbalance
    cap = int(len(phish_df) * 1.2)
    legit_df = legit_df.sample(min(cap, len(legit_df)),
                                random_state=42).reset_index(drop=True)

    merged = (pd.concat([phish_df, legit_df])
                .drop_duplicates(subset="url")
                .sample(frac=1, random_state=42)
                .reset_index(drop=True))

    print(f"\n  Phishing : {(merged['label']==1).sum():>7,}")
    print(f"  Legit    : {(merged['label']==0).sum():>7,}")
    print(f"  Total    : {len(merged):>7,}")
    return merged


# ═══════════════════════════════════════════════════════════════════════════════
#  Live refresh from PhishTank
# ═══════════════════════════════════════════════════════════════════════════════
def refresh_from_phishtank(out_path: Path | None = None) -> pd.Series:
    """
    Download the latest PhishTank verified phishing URLs.
    Requires a free PhishTank account key in env var PHISHTANK_API_KEY,
    or works anonymously (rate-limited).

    Returns a Series of new phishing URLs.
    """
    print("Downloading PhishTank feed...")
    api_key  = os.environ.get("PHISHTANK_API_KEY", "")
    feed_url = (f"{PHISHTANK_URL}?application_key={api_key}"
                if api_key else PHISHTANK_URL)

    try:
        resp = requests.get(feed_url, timeout=60, stream=True)
        resp.raise_for_status()
        raw   = gzip.decompress(resp.content)
        data  = json.loads(raw)
        urls  = pd.Series([entry["url"] for entry in data
                           if entry.get("verified") == "yes"
                           and entry.get("online")   == "yes"])
        print(f"  Downloaded {len(urls):,} verified online phishing URLs")

        if out_path:
            urls.to_frame("url").assign(label=1).to_csv(out_path, index=False)
            print(f"  Saved to {out_path}")

        return urls

    except requests.RequestException as e:
        print(f"  [ERROR] Could not download PhishTank feed: {e}")
        return pd.Series(dtype=str)


# ═══════════════════════════════════════════════════════════════════════════════
#  Dataset statistics
# ═══════════════════════════════════════════════════════════════════════════════
def print_stats(df: pd.DataFrame) -> None:
    total   = len(df)
    phish   = (df["label"] == 1).sum()
    legit   = (df["label"] == 0).sum()
    avg_len = df["url"].str.len().mean()

    print("\n── Dataset Statistics ────────────────────────────────────────────────")
    print(f"  Total URLs : {total:>8,}")
    print(f"  Phishing   : {phish:>8,}  ({phish/total:.1%})")
    print(f"  Legitimate : {legit:>8,}  ({legit/total:.1%})")
    print(f"  Avg URL len: {avg_len:>8.1f} chars")

    # Top-level domain distribution (phishing)
    def extract_tld(url):
        try:
            host = urlparse(url).hostname or ""
            parts = host.split(".")
            return parts[-1] if parts else "?"
        except Exception:
            return "?"

    tlds = df[df["label"] == 1]["url"].apply(extract_tld).value_counts().head(5)
    print("\n  Top TLDs in phishing URLs:")
    for tld, cnt in tlds.items():
        print(f"    .{tld:<8} {cnt:>6,}")
    print("──────────────────────────────────────────────────────────────────────")


# ═══════════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PhishTrace data pipeline")
    parser.add_argument("--refresh", action="store_true",
                        help="Download fresh URLs from PhishTank before merging")
    parser.add_argument("--stats-only", action="store_true",
                        help="Show stats for the existing processed dataset")
    args = parser.parse_args()

    if args.stats_only:
        if DATASET_OUT.exists():
            df = pd.read_csv(DATASET_OUT)
            print_stats(df)
        else:
            print(f"Processed dataset not found at {DATASET_OUT}. Run without --stats-only first.")
        sys.exit(0)

    if args.refresh:
        refresh_from_phishtank(RAW_DIR / "phishtank_fresh.csv")

    df = build_dataset()
    df.to_csv(DATASET_OUT, index=False)
    print(f"\nSaved → {DATASET_OUT}")
    print_stats(df)
    print("\n✅ Data collection complete!")
    print("   Next step: python src/features/extract.py")