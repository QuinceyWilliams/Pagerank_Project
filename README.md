# Pagerank_Project

In this project, you will create a simple search engine for the website <https://www.lawfareblog.com>.
This website provides legal analysis on US national security issues.

**Due date:** Sunday, 18 September at midnight

**Computation:**
This project has low computational requirements.
You are not required to complete it on the lambda server (although you are welcome to if you'd like).

## Background

**Data:**

The `data` folder contains two files that store example "web graphs".
The file `small.csv.gz` contains the example graph from the *Deeper Inside Pagerank* paper.
This is a small graph, so we can manually inspect the contents of this file with the following command:
```
$ zcat data/small.csv.gz
source,target
1,2
1,3
3,1
3,2
3,5
4,5
4,6
5,6
5,4
6,4
```

> **Recall:**
> The `cat` terminal command outputs the contents of a file to stdout, and the `zcat` command first decompressed a gzipped file and then outputs the decompressed contents.

As you can see, the graph is stored as a CSV file.
The first line is a header,
and each subsequent line stores a single edge in the graph.
The first column contains the source node of the edge and the second column the target node.
The file is assumed to be sorted alphabetically.

The second data file `lawfareblog.csv.gz` contains the link structure for the lawfare blog.
Let's take a look at the first 10 of these lines:
```
$ zcat data/lawfareblog.csv.gz | head
source,target
www.lawfareblog.com/,www.lawfareblog.com/topic/interrogation
www.lawfareblog.com/,www.lawfareblog.com/upcoming-events
www.lawfareblog.com/,www.lawfareblog.com/
www.lawfareblog.com/,www.lawfareblog.com/our-comments-policy
www.lawfareblog.com/,www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
www.lawfareblog.com/,www.lawfareblog.com/topic/lawfare-research-paper-series
www.lawfareblog.com/,www.lawfareblog.com/topic/book-reviews
www.lawfareblog.com/,www.lawfareblog.com/documents-related-mueller-investigation
www.lawfareblog.com/,www.lawfareblog.com/topic/international-law-loac
```
You can see that in this file, the node names are URLs.
Semantically, each line corresponds to an HTML `<a>` tag that is contained in the source webpage and links to the target webpage.

We can use the following command to count the total number of links in the file:
```
$ zcat data/lawfareblog.csv.gz | wc -l
1610789
```
Since every link corresponds to a non-zero entry in the `P` matrix,
this is also the value of `nnz(P)`.
(Technically, we should subtract 1 from this value since the `wc -l` command also counts the header line, not just the data lines.)

To get the dimensions of `P`, we need to count the total number of nodes in the graph.
The following command achieves this by: decompressing the file, extracting the first column, removing all duplicate lines, then counting the results.
```
$ zcat data/lawfareblog.csv.gz | cut -f1 -d, | uniq | wc -l
25761
```
This matrix is large enough that computing matrix products for dense matrices takes several minutes on a single CPU.
Fortunately, however, the matrix is very sparse.
The following python code computes the fraction of entries in the matrix with non-zero values:
```
>>> 1610788 / (25760**2)
0.0024274297384360172
```
Thus, by using sparse matrix operations, we will be able to speed up the code significantly.

**Code:**

The `pagerank.py` file contains code for loading the graph CSV files and searching through their nodes for key phrases.
For example, you can perform a search for all nodes (i.e. urls) that mention the string `corona` with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --search_query=corona
```

> **NOTE:**
> It will take about 10 seconds to load and parse the data files.
> All the other computation happens essentially instantly.

Currently, the pagerank of the nodes is not currently being calculated correctly, and so the webpages are returned in an arbitrary order.
Your task in this assignment will be to fix these calculations in order to have the most important results (i.e. highest pagerank results) returned first.

## Task 1: the power method

Implement the `WebGraph.power_method` function in `pagerank.py` for computing the pagerank vector by fixing the `FIXME` annotation.

**Part 1:**

To check that your implementation is working,
you should run the program on the `data/small.csv.gz` graph.
For my implementation, I get the following output.
```
$ python3 pagerank.py --data=data/small.csv.gz --verbose
DEBUG:root:computing indices
DEBUG:root:computing values
DEBUG:root:i=0 residual=2.5629e-01
DEBUG:root:i=1 residual=1.1841e-01
DEBUG:root:i=2 residual=7.0701e-02
DEBUG:root:i=3 residual=3.1815e-02
DEBUG:root:i=4 residual=2.0497e-02
DEBUG:root:i=5 residual=1.0108e-02
DEBUG:root:i=6 residual=6.3716e-03
DEBUG:root:i=7 residual=3.4228e-03
DEBUG:root:i=8 residual=2.0879e-03
DEBUG:root:i=9 residual=1.1750e-03
DEBUG:root:i=10 residual=7.0131e-04
DEBUG:root:i=11 residual=4.0321e-04
DEBUG:root:i=12 residual=2.3800e-04
DEBUG:root:i=13 residual=1.3812e-04
DEBUG:root:i=14 residual=8.1083e-05
DEBUG:root:i=15 residual=4.7251e-05
DEBUG:root:i=16 residual=2.7704e-05
DEBUG:root:i=17 residual=1.6164e-05
DEBUG:root:i=18 residual=9.4778e-06
DEBUG:root:i=19 residual=5.5066e-06
DEBUG:root:i=20 residual=3.2042e-06
DEBUG:root:i=21 residual=1.8612e-06
DEBUG:root:i=22 residual=1.1283e-06
DEBUG:root:i=23 residual=6.1907e-07
INFO:root:rank=0 pagerank=6.6270e-01 url=4
INFO:root:rank=1 pagerank=5.2179e-01 url=6
INFO:root:rank=2 pagerank=4.1434e-01 url=5
INFO:root:rank=3 pagerank=2.3175e-01 url=2
INFO:root:rank=4 pagerank=1.8590e-01 url=3
INFO:root:rank=5 pagerank=1.6917e-01 url=1
```
Yours likely won't be identical (due to weird floating point issues), but it should be similar.
In particular, the ranking of the nodes/urls should be the same order.

> **NOTE:**
> The `--verbose` flag causes all of the lines beginning with `DEBUG` to be printed.
> By default, only lines beginning with `INFO` are printed.

**Part 2:**

The `pagerank.py` file has an option `--search_query`, which takes a string as a parameter.
If this argument is used, then the program returns all nodes that match the query string sorted according to their pagerank.
Essentially, this gives us the most important pages related to our query.

Again, you may not get the exact same results as me,
but you should get similar results to the examples I've shown below.
Verify that you do in fact get similar results.

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
INFO:root:rank=0 pagerank=1.0038e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=1 pagerank=8.9224e-04 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=2 pagerank=7.0390e-04 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=3 pagerank=6.9153e-04 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=4 pagerank=6.7041e-04 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=5 pagerank=6.6256e-04 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
INFO:root:rank=6 pagerank=6.5046e-04 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
INFO:root:rank=7 pagerank=6.3620e-04 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=8 pagerank=6.1248e-04 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus
INFO:root:rank=9 pagerank=6.0187e-04 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
INFO:root:rank=0 pagerank=5.7826e-03 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=5.2338e-03 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
INFO:root:rank=2 pagerank=5.1297e-03 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
INFO:root:rank=3 pagerank=4.6599e-03 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
INFO:root:rank=4 pagerank=4.5934e-03 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
INFO:root:rank=5 pagerank=4.3071e-03 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
INFO:root:rank=6 pagerank=4.0935e-03 url=www.lawfareblog.com/why-trump-cant-buy-greenland
INFO:root:rank=7 pagerank=3.7591e-03 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
INFO:root:rank=8 pagerank=3.4509e-03 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
INFO:root:rank=9 pagerank=3.4484e-03 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
INFO:root:rank=0 pagerank=4.5746e-03 url=www.lawfareblog.com/praise-presidents-iran-tweets
INFO:root:rank=1 pagerank=4.4174e-03 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
INFO:root:rank=2 pagerank=2.6928e-03 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
INFO:root:rank=3 pagerank=1.9391e-03 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
INFO:root:rank=4 pagerank=1.5452e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
INFO:root:rank=5 pagerank=1.5357e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
INFO:root:rank=6 pagerank=1.5258e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
INFO:root:rank=7 pagerank=1.4221e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
INFO:root:rank=8 pagerank=1.1788e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
INFO:root:rank=9 pagerank=1.1463e-03 url=www.lawfareblog.com/israel-iran-syria-clash-and-law-use-force
```

**Part 3:**

The webgraph of lawfareblog.com (i.e. the `P` matrix) naturally contains a lot of structure.
For example, essentially all pages on the domain have links to the root page <https://lawfareblog.com/> and other "non-article" pages like <https://www.lawfareblog.com/topics> and <https://www.lawfareblog.com/subscribe-lawfare>.
These pages therefore have a large pagerank.
We can get a list of the pages with the largest pagerank by running

```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz
INFO:root:rank=0 pagerank=2.8741e-01 url=www.lawfareblog.com/lawfare-job-board
INFO:root:rank=1 pagerank=2.8741e-01 url=www.lawfareblog.com/masthead
INFO:root:rank=2 pagerank=2.8741e-01 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
INFO:root:rank=3 pagerank=2.8741e-01 url=www.lawfareblog.com/documents-related-mueller-investigation
INFO:root:rank=4 pagerank=2.8741e-01 url=www.lawfareblog.com/topics
INFO:root:rank=5 pagerank=2.8741e-01 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
INFO:root:rank=6 pagerank=2.8741e-01 url=www.lawfareblog.com/snowden-revelations
INFO:root:rank=7 pagerank=2.8741e-01 url=www.lawfareblog.com/support-lawfare
INFO:root:rank=8 pagerank=2.8741e-01 url=www.lawfareblog.com/upcoming-events
INFO:root:rank=9 pagerank=2.8741e-01 url=www.lawfareblog.com/our-comments-policy
```

Most of these pages are not very interesting, however, because they are not articles,
and usually when we are performing a web search, we only want articles.

This raises the question: How can we find the most important articles filtering out the non-article pages?
The answer is to modify the `P` matrix by removing all links to non-article pages.

This raises another question: How do we know if a link is a non-article page?
Unfortunately, this is a hard question to answer with 100% accuracy,
but there are many methods that get us most of the way there.
One easy to implement method is to compute what's called the "in-link ratio" of each node (i.e. the total number of edges with the node as a target divided by the total number of nodes),
and then remove nodes from the search results with too-high of a ratio.
The intuition is that non-article pages often appear in the menu of a webpage, and so have links from almost all of the other webpages;
but article-webpages are unlikely to appear on a menu and so will only have a small number of links from other webpages.
The `--filter_ratio` parameter causes the code to remove all pages that have an in-link ratio larger than the provided value.

Using this option, we can estimate the most important articles on the domain with the following command:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull
```
Notice that the urls in this list look much more like articles than the urls in the previous list.

When Google calculates their `P` matrix for the web,
they use a similar (but much more complicated) process to modify the `P` matrix in order to reduce spam results.
The exact formula they use is a jealously guarded secret that they update continuously.

In the case above, notice that we have accidentally removed the blog's most popular article (<www.lawfareblog.com/snowden-revelations>).
The blog editors believed that Snowden's revelations about NSA spying are so important that they directly put a link to the article on the menu.
So every single webpage in the domain links to the Snowden article,
and our "anti-spam" `--filter-ratio` argument removed this article from the list.
In general, it is a challenging open problem to remove spam from pagerank results,
and all current solutions rely on careful human tuning and still have lots of false positives and false negatives.

**Part 4:**

Recall from the reading that the runtime of pagerank depends heavily on the eigengap of the `\bar\bar P` matrix,
and that this eigengap is bounded by the alpha parameter.

Run the following four commands:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose 
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
```
You should notice that the last command takes considerably more iterations to compute the pagerank vector.
(My code takes 685 iterations for this call, and about 10 iterations for all the others.)

This raises the question: Why does the second command (with the `--alpha` option but without the `--filter_ratio`) option not take a long time to run?
The answer is that the `P` graph for <https://www.lawfareblog.com> naturally has a large eigengap and so is fast to compute for all alpha values,
but the modified graph does not have a large eigengap and so requires a small alpha for fast convergence.

Changing the value of alpha also gives us very different pagerank rankings.
For example, 
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
INFO:root:rank=0 pagerank=3.4696e-01 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
INFO:root:rank=1 pagerank=2.9521e-01 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
INFO:root:rank=2 pagerank=2.9040e-01 url=www.lawfareblog.com/opening-statement-david-holmes
INFO:root:rank=3 pagerank=1.5179e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
INFO:root:rank=4 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1963
INFO:root:rank=5 pagerank=1.5099e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1964
INFO:root:rank=6 pagerank=1.5071e-01 url=www.lawfareblog.com/lawfare-podcast-week-was-impeachment
INFO:root:rank=7 pagerank=1.4957e-01 url=www.lawfareblog.com/todays-headlines-and-commentary-1962
INFO:root:rank=8 pagerank=1.4367e-01 url=www.lawfareblog.com/cyberlaw-podcast-mistrusting-google
INFO:root:rank=9 pagerank=1.4240e-01 url=www.lawfareblog.com/lawfare-podcast-bonus-edition-gordon-sondland-vs-committee-no-bull

$ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
INFO:root:rank=0 pagerank=7.0149e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=7.0149e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.0552e-01 url=www.lawfareblog.com/cost-using-zero-days
INFO:root:rank=3 pagerank=3.1755e-02 url=www.lawfareblog.com/lawfare-podcast-former-congressman-brian-baird-and-daniel-schuman-how-congress-can-continue-function
INFO:root:rank=4 pagerank=2.2040e-02 url=www.lawfareblog.com/events
INFO:root:rank=5 pagerank=1.6027e-02 url=www.lawfareblog.com/water-wars-increased-us-focus-indo-pacific
INFO:root:rank=6 pagerank=1.6026e-02 url=www.lawfareblog.com/water-wars-drill-maybe-drill
INFO:root:rank=7 pagerank=1.6023e-02 url=www.lawfareblog.com/water-wars-disjointed-operations-south-china-sea
INFO:root:rank=8 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-song-oil-and-fire
INFO:root:rank=9 pagerank=1.6020e-02 url=www.lawfareblog.com/water-wars-sinking-feeling-philippine-china-relations
```

Which of these rankings is better is entirely subjective,
and the only way to know if you have the "best" alpha for your application is to try several variations and see what is best.
If large alphas are good for your application, you can see that there is a trade-off between quality answers and algorithmic runtime.
We'll be exploring this trade-off more formally in class over the rest of the semester.

## Task 2: the personalization vector

The most interesting applications of pagerank involve the personalization vector.
Implement the `WebGraph.make_personalization_vector` function so that it outputs a personalization vector tuned for the input query.
The pseudocode for the function is:
```
for each index in the personalization vector:
    get the url for the index (see the _index_to_url function)
    check if the url satisfies the input query (see the url_satisfies_query function)
    if so, set the corresponding index to one
normalize the vector
```

**Part 1:**

The command line argument `--personalization_vector_query` will use the function you created above to augment your search with a custom personalization vector.
If you've implemented the function correctly,
you should get results similar to:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=1.2209e-01 url=www.lawfareblog.com/brexit-not-immune-coronavirus
INFO:root:rank=4 pagerank=1.2209e-01 url=www.lawfareblog.com/rational-security-my-corona-edition
INFO:root:rank=5 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=6 pagerank=9.1920e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=7 pagerank=9.1920e-02 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=8 pagerank=7.7770e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=9 pagerank=7.2888e-02 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
```

Notice that these results are significantly different than when using the `--search_query` option:
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --search_query='corona'
INFO:root:rank=0 pagerank=8.1320e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
INFO:root:rank=1 pagerank=7.7908e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
INFO:root:rank=2 pagerank=5.2262e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
INFO:root:rank=3 pagerank=3.9584e-03 url=www.lawfareblog.com/britains-coronavirus-response
INFO:root:rank=4 pagerank=3.8114e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
INFO:root:rank=5 pagerank=3.3973e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
INFO:root:rank=6 pagerank=3.3633e-03 url=www.lawfareblog.com/cyberlaw-podcast-how-israel-fighting-coronavirus
INFO:root:rank=7 pagerank=3.3557e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
INFO:root:rank=8 pagerank=3.2160e-03 url=www.lawfareblog.com/congress-needs-coronavirus-failsafe-its-too-late
INFO:root:rank=9 pagerank=3.1036e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
```

Which results are better?
Again, that depends on what you mean by "better."
With the `--personalization_vector_query` option,
a webpage is important only if other coronavirus webpages also think it's important;
with the `--search_query` option,
a webpage is important if any other webpage thinks it's important.
You'll notice that in the later example, many of the webpages are about Congressional proceedings related to the coronavirus.
From a strictly coronavirus perspective, these are not very important webpages.
But in the broader context of national security, these are very important webpages.

Google engineers spend TONs of time fine-tuning their pagerank personalization vectors to remove spam webpages.
Exactly how they do this is another one of their secrets that they don't publicly talk about.

**Part 2:**

Another use of the `--personalization_vector_query` option is that we can find out what webpages are related to the coronavirus but don't directly mention the coronavirus.
This can be used to map out what types of topics are similar to the coronavirus.

For example, the following query ranks all webpages by their `corona` importance,
but removes webpages mentioning `corona` from the results.
```
$ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
INFO:root:rank=0 pagerank=6.3127e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
INFO:root:rank=1 pagerank=6.3124e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
INFO:root:rank=2 pagerank=1.5947e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
INFO:root:rank=3 pagerank=9.3360e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
INFO:root:rank=4 pagerank=7.0277e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
INFO:root:rank=5 pagerank=6.9713e-02 url=www.lawfareblog.com/lawfare-podcast-mom-and-dad-talk-clinical-trials-pandemic
INFO:root:rank=6 pagerank=6.4944e-02 url=www.lawfareblog.com/limits-world-health-organization
INFO:root:rank=7 pagerank=5.9492e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
INFO:root:rank=8 pagerank=5.1245e-02 url=www.lawfareblog.com/us-moves-dismiss-case-against-company-linked-ira-troll-farm
INFO:root:rank=9 pagerank=5.1245e-02 url=www.lawfareblog.com/livestream-house-armed-services-holds-hearing-national-security-challenges-north-and-south-america
```
You can see that there are many urls about concepts that are obviously related like "covid", "clinical trials", and "quarantine",
but this algorithm also finds articles about Chinese propaganda and Trump's policy decisions.
Both of these articles are highly relevant to coronavirus discussions,
but a simple keyword search for corona or related terms would not find these articles.

<!--
**Part 3:**

Select another topic related to national security.
You should experiment with a national security topic other than the coronavirus.
For example, find out what articles are important to the `iran` topic but do not contain the word `iran`.
Your goal should be to discover what topics that www.lawfareblog.com considers to be related to the national security topic you choose.
-->

## Submission

1. Create a new repo on github (not a fork of this repo).

2. Run the following commands, and paste their output into the code blocks below.
   
   Task 1, part 1:
   ```
   $ python3 pagerank.py --data=data/small.csv.gz --verbose
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=0.36601030826568604
    DEBUG:root:i=1 residual=0.18730977177619934
    DEBUG:root:i=2 residual=0.07537698745727539
    DEBUG:root:i=3 residual=0.04784689471125603
    DEBUG:root:i=4 residual=0.02921396680176258
    DEBUG:root:i=5 residual=0.022731073200702667
    DEBUG:root:i=6 residual=0.015631210058927536
    DEBUG:root:i=7 residual=0.0116210225969553
    DEBUG:root:i=8 residual=0.008277525193989277
    DEBUG:root:i=9 residual=0.0060295723378658295
    DEBUG:root:i=10 residual=0.004339270759373903
    DEBUG:root:i=11 residual=0.003141967346891761
    DEBUG:root:i=12 residual=0.0022676654625684023
    DEBUG:root:i=13 residual=0.0016393298283219337
    DEBUG:root:i=14 residual=0.0011840679217129946
    DEBUG:root:i=15 residual=0.000855624268297106
    DEBUG:root:i=16 residual=0.0006181371281854808
    DEBUG:root:i=17 residual=0.0004466171667445451
    DEBUG:root:i=18 residual=0.0003227185516152531
    DEBUG:root:i=19 residual=0.00023311786935664713
    DEBUG:root:i=20 residual=0.00016844022320583463
    DEBUG:root:i=21 residual=0.00012171261187177151
    DEBUG:root:i=22 residual=8.79266153788194e-05
    DEBUG:root:i=23 residual=6.350992771331221e-05
    DEBUG:root:i=24 residual=4.590025855577551e-05
    DEBUG:root:i=25 residual=3.3155109122162685e-05
    DEBUG:root:i=26 residual=2.397396383457817e-05
    DEBUG:root:i=27 residual=1.7324668078799732e-05
    DEBUG:root:i=28 residual=1.24961316032568e-05
    DEBUG:root:i=29 residual=9.023437087307684e-06
    DEBUG:root:i=30 residual=6.513716471090447e-06
    DEBUG:root:i=31 residual=4.715175691671902e-06
    DEBUG:root:i=32 residual=3.4036027045658557e-06
    DEBUG:root:i=33 residual=2.473951781212236e-06
    DEBUG:root:i=34 residual=1.7958835769604775e-06
    DEBUG:root:i=35 residual=1.2839233249906101e-06
    DEBUG:root:i=36 residual=9.269911629417038e-07
    INFO:root:rank=0 pagerank=3.0585e-01 url=4
    INFO:root:rank=1 pagerank=2.4170e-01 url=6
    INFO:root:rank=2 pagerank=1.9371e-01 url=5
    INFO:root:rank=3 pagerank=1.1329e-01 url=2
    INFO:root:rank=4 pagerank=9.1303e-02 url=3
    INFO:root:rank=5 pagerank=8.3226e-02 url=1
   ```

   Task 1, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='corona'
    INFO:root:rank=0 pagerank=3.9232e-03 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
    INFO:root:rank=1 pagerank=3.4356e-03 url=www.lawfareblog.com/house-oversight-committee-holds-day-two-hearing-government-coronavirus-response
    INFO:root:rank=2 pagerank=2.3215e-03 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=3 pagerank=2.2608e-03 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
    INFO:root:rank=4 pagerank=2.1224e-03 url=www.lawfareblog.com/israeli-emergency-regulations-location-tracking-coronavirus-carriers
    INFO:root:rank=5 pagerank=2.0719e-03 url=www.lawfareblog.com/why-congress-conducting-business-usual-face-coronavirus
    INFO:root:rank=6 pagerank=2.0331e-03 url=www.lawfareblog.com/congressional-homeland-security-committees-seek-ways-support-state-federal-responses-coronavirus
    INFO:root:rank=7 pagerank=1.9727e-03 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
    INFO:root:rank=8 pagerank=1.9692e-03 url=www.lawfareblog.com/livestream-house-oversight-committee-holds-hearing-government-coronavirus-response
    INFO:root:rank=9 pagerank=1.8278e-03 url=www.lawfareblog.com/house-subcommittee-voices-concerns-over-us-management-coronavirus

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='trump'
    INFO:root:rank=0 pagerank=4.2797e-02 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=3.4680e-02 url=www.lawfareblog.com/donald-trump-and-politically-weaponized-executive-branch
    INFO:root:rank=2 pagerank=2.7846e-02 url=www.lawfareblog.com/trump-administrations-worrying-new-policy-israeli-settlements
    INFO:root:rank=3 pagerank=2.6348e-02 url=www.lawfareblog.com/document-trump-revokes-obama-executive-order-counterterrorism-strike-casualty-reporting
    INFO:root:rank=4 pagerank=2.4783e-02 url=www.lawfareblog.com/dc-circuit-overrules-district-courts-due-process-ruling-qasim-v-trump
    INFO:root:rank=5 pagerank=2.2740e-02 url=www.lawfareblog.com/how-trumps-approach-middle-east-ignores-past-future-and-human-condition
    INFO:root:rank=6 pagerank=2.0580e-02 url=www.lawfareblog.com/why-trump-cant-buy-greenland
    INFO:root:rank=7 pagerank=1.8465e-02 url=www.lawfareblog.com/oral-argument-summary-qassim-v-trump
    INFO:root:rank=8 pagerank=1.7371e-02 url=www.lawfareblog.com/dc-circuit-court-denies-trump-rehearing-mazars-case
    INFO:root:rank=9 pagerank=1.7179e-02 url=www.lawfareblog.com/second-circuit-rules-mazars-must-hand-over-trump-tax-returns-new-york-prosecutors

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --search_query='iran'
    INFO:root:rank=0 pagerank=3.4587e-02 url=www.lawfareblog.com/praise-presidents-iran-tweets
    INFO:root:rank=1 pagerank=2.3336e-02 url=www.lawfareblog.com/how-us-iran-tensions-could-disrupt-iraqs-fragile-peace
    INFO:root:rank=2 pagerank=1.4031e-02 url=www.lawfareblog.com/cyber-command-operational-update-clarifying-june-2019-iran-operation
    INFO:root:rank=3 pagerank=1.1127e-02 url=www.lawfareblog.com/aborted-iran-strike-fine-line-between-necessity-and-revenge
    INFO:root:rank=4 pagerank=6.9524e-03 url=www.lawfareblog.com/iranian-hostage-crisis-and-its-effect-american-politics
    INFO:root:rank=5 pagerank=6.9314e-03 url=www.lawfareblog.com/parsing-state-departments-letter-use-force-against-iran
    INFO:root:rank=6 pagerank=6.8205e-03 url=www.lawfareblog.com/announcing-united-states-and-use-force-against-iran-new-lawfare-e-book
    INFO:root:rank=7 pagerank=6.0691e-03 url=www.lawfareblog.com/us-names-iranian-revolutionary-guard-terrorist-organization-and-sanctions-international-criminal
    INFO:root:rank=8 pagerank=5.8297e-03 url=www.lawfareblog.com/trump-moves-cut-irans-oil-revenues-whats-his-endgame
    INFO:root:rank=9 pagerank=4.9607e-03 url=www.lawfareblog.com/iran-shoots-down-us-drone-domestic-and-international-legal-implications
   ```

   Task 1, part 3:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz
    INFO:root:rank=0 pagerank=3.6344e+00 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=1 pagerank=3.6344e+00 url=www.lawfareblog.com/masthead
    INFO:root:rank=2 pagerank=3.6344e+00 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=3 pagerank=3.6344e+00 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=4 pagerank=3.6344e+00 url=www.lawfareblog.com/topics
    INFO:root:rank=5 pagerank=3.6344e+00 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
    INFO:root:rank=6 pagerank=3.6344e+00 url=www.lawfareblog.com/snowden-revelations
    INFO:root:rank=7 pagerank=3.6344e+00 url=www.lawfareblog.com/support-lawfare
    INFO:root:rank=8 pagerank=3.6344e+00 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=3.6344e+00 url=www.lawfareblog.com/our-comments-policy

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2
    INFO:root:rank=0 pagerank=1.4522e+00 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=1.1105e+00 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=1.0983e+00 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=5.2522e-01 url=www.lawfareblog.com/summary-david-holmess-deposition-testimony
    INFO:root:rank=4 pagerank=5.2238e-01 url=www.lawfareblog.com/senate-examines-threats-homeland
    INFO:root:rank=5 pagerank=4.5228e-01 url=www.lawfareblog.com/what-make-first-day-impeachment-hearings
    INFO:root:rank=6 pagerank=4.5181e-01 url=www.lawfareblog.com/livestream-house-armed-services-committee-hearing-f-35-program
    INFO:root:rank=7 pagerank=4.4813e-01 url=www.lawfareblog.com/whats-house-resolution-impeachment
    INFO:root:rank=8 pagerank=4.1616e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
    INFO:root:rank=9 pagerank=4.1559e-01 url=www.lawfareblog.com/congress-us-policy-toward-syria-and-turkey-overview-recent-hearings
   ```

   Task 1, part 4:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=17.438602447509766
    DEBUG:root:i=1 residual=2.351188898086548
    DEBUG:root:i=2 residual=1.0328168869018555
    DEBUG:root:i=3 residual=1.407150387763977
    DEBUG:root:i=4 residual=1.193509817123413
    DEBUG:root:i=5 residual=0.9074628353118896
    DEBUG:root:i=6 residual=0.6669260263442993
    DEBUG:root:i=7 residual=0.4843292236328125
    DEBUG:root:i=8 residual=0.3507104218006134
    DEBUG:root:i=9 residual=0.2535938620567322
    DEBUG:root:i=10 residual=0.18325957655906677
    DEBUG:root:i=11 residual=0.13242053985595703
    DEBUG:root:i=12 residual=0.09567706286907196
    DEBUG:root:i=13 residual=0.06912738084793091
    DEBUG:root:i=14 residual=0.04994602128863335
    DEBUG:root:i=15 residual=0.03608503192663193
    DEBUG:root:i=16 residual=0.026072191074490547
    DEBUG:root:i=17 residual=0.018836021423339844
    DEBUG:root:i=18 residual=0.013611502945423126
    DEBUG:root:i=19 residual=0.009834385477006435
    DEBUG:root:i=20 residual=0.007107291370630264
    DEBUG:root:i=21 residual=0.005132816731929779
    DEBUG:root:i=22 residual=0.0037102000787854195
    DEBUG:root:i=23 residual=0.0026791770942509174
    DEBUG:root:i=24 residual=0.0019339974969625473
    DEBUG:root:i=25 residual=0.0013970049330964684
    DEBUG:root:i=26 residual=0.0010103709064424038
    DEBUG:root:i=27 residual=0.0007311342633329332
    DEBUG:root:i=28 residual=0.0005270783440209925
    DEBUG:root:i=29 residual=0.0003791998606175184
    DEBUG:root:i=30 residual=0.00027510474319569767
    DEBUG:root:i=31 residual=0.00019827399228233844
    DEBUG:root:i=32 residual=0.0001437486644135788
    DEBUG:root:i=33 residual=0.00010326805204385892
    DEBUG:root:i=34 residual=7.352694956352934e-05
    DEBUG:root:i=35 residual=5.617599526885897e-05
    DEBUG:root:i=36 residual=3.965529322158545e-05
    DEBUG:root:i=37 residual=2.8914744689245708e-05
    DEBUG:root:i=38 residual=2.065368607873097e-05
    DEBUG:root:i=39 residual=1.3219072570791468e-05
    DEBUG:root:i=40 residual=9.088201295526233e-06
    DEBUG:root:i=41 residual=8.260468348453287e-06
    DEBUG:root:i=42 residual=7.4343893174955156e-06
    DEBUG:root:i=43 residual=2.4806408873701002e-06
    DEBUG:root:i=44 residual=1.652934770390857e-06
    DEBUG:root:i=45 residual=2.4779344585112995e-06
    DEBUG:root:i=46 residual=3.3037613320630044e-06
    DEBUG:root:i=47 residual=1.652887362979527e-06
    DEBUG:root:i=48 residual=8.260826120931597e-07
    INFO:root:rank=0 pagerank=3.6344e+00 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=1 pagerank=3.6344e+00 url=www.lawfareblog.com/masthead
    INFO:root:rank=2 pagerank=3.6344e+00 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=3 pagerank=3.6344e+00 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=4 pagerank=3.6344e+00 url=www.lawfareblog.com/topics
    INFO:root:rank=5 pagerank=3.6344e+00 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
    INFO:root:rank=6 pagerank=3.6344e+00 url=www.lawfareblog.com/snowden-revelations
    INFO:root:rank=7 pagerank=3.6344e+00 url=www.lawfareblog.com/support-lawfare
    INFO:root:rank=8 pagerank=3.6344e+00 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=3.6344e+00 url=www.lawfareblog.com/our-comments-policy
   
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --alpha=0.99999
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=20.524179458618164
    DEBUG:root:i=1 residual=3.253918170928955
    DEBUG:root:i=2 residual=1.682097315788269
    DEBUG:root:i=3 residual=2.69679856300354
    DEBUG:root:i=4 residual=2.692432403564453
    DEBUG:root:i=5 residual=2.4082930088043213
    DEBUG:root:i=6 residual=2.0822160243988037
    DEBUG:root:i=7 residual=1.780108094215393
    DEBUG:root:i=8 residual=1.5160757303237915
    DEBUG:root:i=9 residual=1.2895264625549316
    DEBUG:root:i=10 residual=1.0963314771652222
    DEBUG:root:i=11 residual=0.931946873664856
    DEBUG:root:i=12 residual=0.7921083569526672
    DEBUG:root:i=13 residual=0.6732878088951111
    DEBUG:root:i=14 residual=0.5722880959510803
    DEBUG:root:i=15 residual=0.4864388108253479
    DEBUG:root:i=16 residual=0.41343215107917786
    DEBUG:root:i=17 residual=0.35141366720199585
    DEBUG:root:i=18 residual=0.2986777424812317
    DEBUG:root:i=19 residual=0.25387269258499146
    DEBUG:root:i=20 residual=0.21581102907657623
    DEBUG:root:i=21 residual=0.18343135714530945
    DEBUG:root:i=22 residual=0.15591321885585785
    DEBUG:root:i=23 residual=0.132499098777771
    DEBUG:root:i=24 residual=0.11257253587245941
    DEBUG:root:i=25 residual=0.09568384289741516
    DEBUG:root:i=26 residual=0.08129722625017166
    DEBUG:root:i=27 residual=0.06907970458269119
    DEBUG:root:i=28 residual=0.058730609714984894
    DEBUG:root:i=29 residual=0.049924302846193314
    DEBUG:root:i=30 residual=0.042432352900505066
    DEBUG:root:i=31 residual=0.036081086844205856
    DEBUG:root:i=32 residual=0.030668381601572037
    DEBUG:root:i=33 residual=0.026067443192005157
    DEBUG:root:i=34 residual=0.022158129140734673
    DEBUG:root:i=35 residual=0.018834112212061882
    DEBUG:root:i=36 residual=0.01600641943514347
    DEBUG:root:i=37 residual=0.013605199754238129
    DEBUG:root:i=38 residual=0.011564274318516254
    DEBUG:root:i=39 residual=0.009830532595515251
    DEBUG:root:i=40 residual=0.008355554193258286
    DEBUG:root:i=41 residual=0.007101733237504959
    DEBUG:root:i=42 residual=0.006036014761775732
    DEBUG:root:i=43 residual=0.005130234640091658
    DEBUG:root:i=44 residual=0.0043603540398180485
    DEBUG:root:i=45 residual=0.00370638445019722
    DEBUG:root:i=46 residual=0.0031504319049417973
    DEBUG:root:i=47 residual=0.002677850192412734
    DEBUG:root:i=48 residual=0.002276252955198288
    DEBUG:root:i=49 residual=0.001934896339662373
    DEBUG:root:i=50 residual=0.0016444281209260225
    DEBUG:root:i=51 residual=0.0013977519702166319
    DEBUG:root:i=52 residual=0.0011880815727636218
    DEBUG:root:i=53 residual=0.0010095390025526285
    DEBUG:root:i=54 residual=0.000858052633702755
    DEBUG:root:i=55 residual=0.0007290812209248543
    DEBUG:root:i=56 residual=0.0006196906906552613
    DEBUG:root:i=57 residual=0.0005267122178338468
    DEBUG:root:i=58 residual=0.0004476935137063265
    DEBUG:root:i=59 residual=0.00038053691969253123
    DEBUG:root:i=60 residual=0.0003233947791159153
    DEBUG:root:i=61 residual=0.0002748775586951524
    DEBUG:root:i=62 residual=0.000233659622608684
    DEBUG:root:i=63 residual=0.0001986084971576929
    DEBUG:root:i=64 residual=0.00016881860210560262
    DEBUG:root:i=65 residual=0.00014354403538163751
    DEBUG:root:i=66 residual=0.0001220137783093378
    DEBUG:root:i=67 residual=0.00010371220560045913
    DEBUG:root:i=68 residual=8.817081106826663e-05
    DEBUG:root:i=69 residual=7.494266901630908e-05
    DEBUG:root:i=70 residual=6.3699008023832e-05
    DEBUG:root:i=71 residual=5.414391853264533e-05
    DEBUG:root:i=72 residual=4.602106855600141e-05
    DEBUG:root:i=73 residual=3.911665771738626e-05
    DEBUG:root:i=74 residual=3.3247750252485275e-05
    DEBUG:root:i=75 residual=2.825964111252688e-05
    DEBUG:root:i=76 residual=2.402102472842671e-05
    DEBUG:root:i=77 residual=2.0424202375579625e-05
    DEBUG:root:i=78 residual=1.7418315110262483e-05
    DEBUG:root:i=79 residual=1.4840066796750762e-05
    DEBUG:root:i=80 residual=1.2625672752619721e-05
    DEBUG:root:i=81 residual=1.073116345651215e-05
    DEBUG:root:i=82 residual=9.122042683884501e-06
    DEBUG:root:i=83 residual=7.753747013339307e-06
    DEBUG:root:i=84 residual=6.590776138182264e-06
    DEBUG:root:i=85 residual=5.602069904853124e-06
    DEBUG:root:i=86 residual=4.761406216857722e-06
    DEBUG:root:i=87 residual=4.04720503865974e-06
    DEBUG:root:i=88 residual=3.4405070437060203e-06
    DEBUG:root:i=89 residual=2.9243708468129626e-06
    DEBUG:root:i=90 residual=2.485684944986133e-06
    DEBUG:root:i=91 residual=2.1125501916685607e-06
    DEBUG:root:i=92 residual=1.7958891476155259e-06
    DEBUG:root:i=93 residual=1.5264250805557822e-06
    DEBUG:root:i=94 residual=1.2975017398275668e-06
    DEBUG:root:i=95 residual=1.1028663493561908e-06
    DEBUG:root:i=96 residual=9.37476670515025e-07
    INFO:root:rank=0 pagerank=5.6268e-04 url=www.lawfareblog.com/lawfare-job-board
    INFO:root:rank=1 pagerank=5.6268e-04 url=www.lawfareblog.com/masthead
    INFO:root:rank=2 pagerank=5.6268e-04 url=www.lawfareblog.com/litigation-documents-related-appointment-matthew-whitaker-acting-attorney-general
    INFO:root:rank=3 pagerank=5.6268e-04 url=www.lawfareblog.com/documents-related-mueller-investigation
    INFO:root:rank=4 pagerank=5.6268e-04 url=www.lawfareblog.com/topics
    INFO:root:rank=5 pagerank=5.6268e-04 url=www.lawfareblog.com/about-lawfare-brief-history-term-and-site
    INFO:root:rank=6 pagerank=5.6268e-04 url=www.lawfareblog.com/snowden-revelations
    INFO:root:rank=7 pagerank=5.6268e-04 url=www.lawfareblog.com/support-lawfare
    INFO:root:rank=8 pagerank=5.6268e-04 url=www.lawfareblog.com/upcoming-events
    INFO:root:rank=9 pagerank=5.6268e-04 url=www.lawfareblog.com/our-comments-policy

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=4.351032733917236
    DEBUG:root:i=1 residual=2.267399549484253
    DEBUG:root:i=2 residual=1.287729263305664
    DEBUG:root:i=3 residual=0.6398324370384216
    DEBUG:root:i=4 residual=0.362384557723999
    DEBUG:root:i=5 residual=0.28065118193626404
    DEBUG:root:i=6 residual=0.23211678862571716
    DEBUG:root:i=7 residual=0.18454837799072266
    DEBUG:root:i=8 residual=0.14152319729328156
    DEBUG:root:i=9 residual=0.10613381117582321
    DEBUG:root:i=10 residual=0.07853900641202927
    DEBUG:root:i=11 residual=0.05761691927909851
    DEBUG:root:i=12 residual=0.04199717193841934
    DEBUG:root:i=13 residual=0.030449487268924713
    DEBUG:root:i=14 residual=0.021973280236124992
    DEBUG:root:i=15 residual=0.015790091827511787
    DEBUG:root:i=16 residual=0.011306853033602238
    DEBUG:root:i=17 residual=0.00807281769812107
    DEBUG:root:i=18 residual=0.005751962773501873
    DEBUG:root:i=19 residual=0.004093720577657223
    DEBUG:root:i=20 residual=0.00291347480379045
    DEBUG:root:i=21 residual=0.002076070988550782
    DEBUG:root:i=22 residual=0.0014828951098024845
    DEBUG:root:i=23 residual=0.0010632997145876288
    DEBUG:root:i=24 residual=0.0007659068796783686
    DEBUG:root:i=25 residual=0.0005545806488953531
    DEBUG:root:i=26 residual=0.0004038390179630369
    DEBUG:root:i=27 residual=0.0002958480326924473
    DEBUG:root:i=28 residual=0.00021785532590001822
    DEBUG:root:i=29 residual=0.00016123826208058745
    DEBUG:root:i=30 residual=0.00011982970318058506
    DEBUG:root:i=31 residual=8.920086838770658e-05
    DEBUG:root:i=32 residual=6.674823816865683e-05
    DEBUG:root:i=33 residual=4.988350337953307e-05
    DEBUG:root:i=34 residual=3.7340047128964216e-05
    DEBUG:root:i=35 residual=2.7988591682515107e-05
    DEBUG:root:i=36 residual=2.0950883481418714e-05
    DEBUG:root:i=37 residual=1.568375591887161e-05
    DEBUG:root:i=38 residual=1.1711344996001571e-05
    DEBUG:root:i=39 residual=8.744077604205813e-06
    DEBUG:root:i=40 residual=6.5373747020203155e-06
    DEBUG:root:i=41 residual=4.847868240176467e-06
    DEBUG:root:i=42 residual=3.6425774396775523e-06
    DEBUG:root:i=43 residual=2.709269665501779e-06
    DEBUG:root:i=44 residual=2.043208496615989e-06
    DEBUG:root:i=45 residual=1.4751235539733898e-06
    DEBUG:root:i=46 residual=1.100692429645278e-06
    DEBUG:root:i=47 residual=8.054188924688788e-07
    INFO:root:rank=0 pagerank=1.4522e+00 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=1.1105e+00 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=1.0983e+00 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=5.2522e-01 url=www.lawfareblog.com/summary-david-holmess-deposition-testimony
    INFO:root:rank=4 pagerank=5.2238e-01 url=www.lawfareblog.com/senate-examines-threats-homeland
    INFO:root:rank=5 pagerank=4.5228e-01 url=www.lawfareblog.com/what-make-first-day-impeachment-hearings
    INFO:root:rank=6 pagerank=4.5181e-01 url=www.lawfareblog.com/livestream-house-armed-services-committee-hearing-f-35-program
    INFO:root:rank=7 pagerank=4.4813e-01 url=www.lawfareblog.com/whats-house-resolution-impeachment
    INFO:root:rank=8 pagerank=4.1616e-01 url=www.lawfareblog.com/lawfare-podcast-ben-nimmo-whack-mole-game-disinformation
    INFO:root:rank=9 pagerank=4.1559e-01 url=www.lawfareblog.com/congress-us-policy-toward-syria-and-turkey-overview-recent-hearings

   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --verbose --filter_ratio=0.2 --alpha=0.99999
    DEBUG:root:computing indices
    DEBUG:root:computing values
    DEBUG:root:i=0 residual=5.118829250335693
    DEBUG:root:i=1 residual=3.1382076740264893
    DEBUG:root:i=2 residual=2.0967957973480225
    DEBUG:root:i=3 residual=1.225667119026184
    DEBUG:root:i=4 residual=0.8166513442993164
    DEBUG:root:i=5 residual=0.7440460920333862
    DEBUG:root:i=6 residual=0.7239564061164856
    DEBUG:root:i=7 residual=0.6771609783172607
    DEBUG:root:i=8 residual=0.6109223961830139
    DEBUG:root:i=9 residual=0.539002537727356
    DEBUG:root:i=10 residual=0.4692438840866089
    DEBUG:root:i=11 residual=0.4049862325191498
    DEBUG:root:i=12 residual=0.3472883999347687
    DEBUG:root:i=13 residual=0.29622602462768555
    DEBUG:root:i=14 residual=0.2514875531196594
    DEBUG:root:i=15 residual=0.2126174420118332
    DEBUG:root:i=16 residual=0.17910879850387573
    DEBUG:root:i=17 residual=0.15044313669204712
    DEBUG:root:i=18 residual=0.1261032074689865
    DEBUG:root:i=19 residual=0.10558569431304932
    DEBUG:root:i=20 residual=0.08840801566839218
    DEBUG:root:i=21 residual=0.07411528378725052
    DEBUG:root:i=22 residual=0.06228560209274292
    DEBUG:root:i=23 residual=0.05253390222787857
    DEBUG:root:i=24 residual=0.04451422393321991
    DEBUG:root:i=25 residual=0.03792119026184082
    DEBUG:root:i=26 residual=0.03249073401093483
    DEBUG:root:i=27 residual=0.0279986709356308
    DEBUG:root:i=28 residual=0.024258682504296303
    DEBUG:root:i=29 residual=0.02111946977674961
    DEBUG:root:i=30 residual=0.018460439518094063
    DEBUG:root:i=31 residual=0.01618700847029686
    DEBUG:root:i=32 residual=0.01422611903399229
    DEBUG:root:i=33 residual=0.012521550990641117
    DEBUG:root:i=34 residual=0.011030278168618679
    DEBUG:root:i=35 residual=0.009719058871269226
    DEBUG:root:i=36 residual=0.008562091737985611
    DEBUG:root:i=37 residual=0.007538752164691687
    DEBUG:root:i=38 residual=0.0066325003281235695
    DEBUG:root:i=39 residual=0.005829503759741783
    DEBUG:root:i=40 residual=0.005118175875395536
    DEBUG:root:i=41 residual=0.004488406237214804
    DEBUG:root:i=42 residual=0.003931439947336912
    DEBUG:root:i=43 residual=0.0034394152462482452
    DEBUG:root:i=44 residual=0.003005384234711528
    DEBUG:root:i=45 residual=0.0026230397634208202
    DEBUG:root:i=46 residual=0.0022867457009851933
    DEBUG:root:i=47 residual=0.0019913853611797094
    DEBUG:root:i=48 residual=0.0017323445063084364
    DEBUG:root:i=49 residual=0.0015054959803819656
    DEBUG:root:i=50 residual=0.0013071016874164343
    DEBUG:root:i=51 residual=0.0011338291224092245
    DEBUG:root:i=52 residual=0.0009826816385611892
    DEBUG:root:i=53 residual=0.0008509874460287392
    DEBUG:root:i=54 residual=0.0007363799959421158
    DEBUG:root:i=55 residual=0.0006367416935972869
    DEBUG:root:i=56 residual=0.0005502094863913953
    DEBUG:root:i=57 residual=0.00047512789024040103
    DEBUG:root:i=58 residual=0.0004100415389984846
    DEBUG:root:i=59 residual=0.0003536648291628808
    DEBUG:root:i=60 residual=0.0003048755170311779
    DEBUG:root:i=61 residual=0.00026267857174389064
    DEBUG:root:i=62 residual=0.00022621349489782006
    DEBUG:root:i=63 residual=0.00019471911946311593
    DEBUG:root:i=64 residual=0.0001675371895544231
    DEBUG:root:i=65 residual=0.00014409002324100584
    DEBUG:root:i=66 residual=0.00012387441529426724
    DEBUG:root:i=67 residual=0.00010645692964317277
    DEBUG:root:i=68 residual=9.145508374786004e-05
    DEBUG:root:i=69 residual=7.854167051846161e-05
    DEBUG:root:i=70 residual=6.742940604453906e-05
    DEBUG:root:i=71 residual=5.787223926745355e-05
    DEBUG:root:i=72 residual=4.965497282682918e-05
    DEBUG:root:i=73 residual=4.259299021214247e-05
    DEBUG:root:i=74 residual=3.652556915767491e-05
    DEBUG:root:i=75 residual=3.131463745376095e-05
    DEBUG:root:i=76 residual=2.6840840291697532e-05
    DEBUG:root:i=77 residual=2.3000824512564577e-05
    DEBUG:root:i=78 residual=1.9705963495653123e-05
    DEBUG:root:i=79 residual=1.6879481336218305e-05
    DEBUG:root:i=80 residual=1.4455574273597449e-05
    DEBUG:root:i=81 residual=1.2377334314805921e-05
    DEBUG:root:i=82 residual=1.0595984349492937e-05
    DEBUG:root:i=83 residual=9.069332918443251e-06
    DEBUG:root:i=84 residual=7.761332199152093e-06
    DEBUG:root:i=85 residual=6.640925221290672e-06
    DEBUG:root:i=86 residual=5.681349193764618e-06
    DEBUG:root:i=87 residual=4.859711680182954e-06
    DEBUG:root:i=88 residual=4.156278464506613e-06
    DEBUG:root:i=89 residual=3.5541852412279695e-06
    DEBUG:root:i=90 residual=3.0388985123863677e-06
    DEBUG:root:i=91 residual=2.597981620056089e-06
    DEBUG:root:i=92 residual=2.2207473193702754e-06
    DEBUG:root:i=93 residual=1.8980715594807407e-06
    DEBUG:root:i=94 residual=1.622090508135443e-06
    DEBUG:root:i=95 residual=1.3860818626199034e-06
    DEBUG:root:i=96 residual=1.1842829508168506e-06
    DEBUG:root:i=97 residual=1.0117538522536051e-06
    DEBUG:root:i=98 residual=8.642730335850501e-07
    INFO:root:rank=0 pagerank=2.8524e-04 url=www.lawfareblog.com/trump-asks-supreme-court-stay-congressional-subpeona-tax-returns
    INFO:root:rank=1 pagerank=1.8480e-04 url=www.lawfareblog.com/livestream-nov-21-impeachment-hearings-0
    INFO:root:rank=2 pagerank=1.8357e-04 url=www.lawfareblog.com/opening-statement-david-holmes
    INFO:root:rank=3 pagerank=1.2487e-04 url=www.lawfareblog.com/senate-examines-threats-homeland
    INFO:root:rank=4 pagerank=1.1618e-04 url=www.lawfareblog.com/what-make-first-day-impeachment-hearings
    INFO:root:rank=5 pagerank=1.1613e-04 url=www.lawfareblog.com/livestream-house-armed-services-committee-hearing-f-35-program
    INFO:root:rank=6 pagerank=1.1571e-04 url=www.lawfareblog.com/whats-house-resolution-impeachment
    INFO:root:rank=7 pagerank=1.0928e-04 url=www.lawfareblog.com/congress-us-policy-toward-syria-and-turkey-overview-recent-hearings
    INFO:root:rank=8 pagerank=1.0399e-04 url=www.lawfareblog.com/summary-david-holmess-deposition-testimony
    INFO:root:rank=9 pagerank=6.0985e-05 url=www.lawfareblog.com/events
   ```

   Task 2, part 1:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona'
    INFO:root:rank=0 pagerank=2.4868e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=2.4866e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.1435e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=8.0579e-02 url=www.lawfareblog.com/rational-security-my-corona-edition
    INFO:root:rank=4 pagerank=8.0579e-02 url=www.lawfareblog.com/brexit-not-immune-coronavirus
    INFO:root:rank=5 pagerank=7.4815e-02 url=www.lawfareblog.com/prosecuting-purposeful-coronavirus-exposure-terrorism
    INFO:root:rank=6 pagerank=7.4815e-02 url=www.lawfareblog.com/britains-coronavirus-response
    INFO:root:rank=7 pagerank=6.6662e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=8 pagerank=6.4483e-02 url=www.lawfareblog.com/lawfare-podcast-united-nations-and-coronavirus-crisis
    INFO:root:rank=9 pagerank=5.6433e-02 url=www.lawfareblog.com/paper-hearing-experts-debate-digital-contact-tracing-and-coronavirus-privacy-concerns
   ```

   Task 2, part 2:
   ```
   $ python3 pagerank.py --data=data/lawfareblog.csv.gz --filter_ratio=0.2 --personalization_vector_query='corona' --search_query='-corona'
    INFO:root:rank=0 pagerank=2.4868e-01 url=www.lawfareblog.com/covid-19-speech-and-surveillance-response
    INFO:root:rank=1 pagerank=2.4866e-01 url=www.lawfareblog.com/lawfare-live-covid-19-speech-and-surveillance
    INFO:root:rank=2 pagerank=1.1435e-01 url=www.lawfareblog.com/chinatalk-how-party-takes-its-propaganda-global
    INFO:root:rank=3 pagerank=6.6662e-02 url=www.lawfareblog.com/trump-cant-reopen-country-over-state-objections
    INFO:root:rank=4 pagerank=5.2454e-02 url=www.lawfareblog.com/fault-lines-foreign-policy-quarantined
    INFO:root:rank=5 pagerank=4.4881e-02 url=www.lawfareblog.com/limits-world-health-organization
    INFO:root:rank=6 pagerank=4.2900e-02 url=www.lawfareblog.com/chinatalk-dispatches-shanghai-beijing-and-hong-kong
    INFO:root:rank=7 pagerank=3.6023e-02 url=www.lawfareblog.com/trump-cant-play-politics-aid-states
    INFO:root:rank=8 pagerank=3.4500e-02 url=www.lawfareblog.com/senators-urge-cyber-leaders-prevent-attacks-healthcare-sector
    INFO:root:rank=9 pagerank=3.4413e-02 url=www.lawfareblog.com/how-do-you-spy-when-world-shut-down
   ```

3. Ensure that all your changes to the `pagerank.py` and `README.md` files are committed to your repo and pushed to github.

4. Get at least 5 stars on your repo.
   (You may trade stars with other students in the class.)

   > **NOTE:**
   > 
   > Recruiters use github profiles to determine who to hire,
   > and pagerank is used to rank user profiles and projects.
   > Links in this graph correspond to who has starred/followed who's repo.
   > By getting more stars on your repo, you'll be increasing your github pagerank, which increases the likelihood that recruiters will hire you.
   > To see an example, [perform a search for `data mining`](https://github.com/search?q=data+mining).
   > Notice that the results are returned "approximately" ranked by the number of stars,
   > but because "some stars count more than others" the results are not exactly ranked by the number of stars.
   > (I asked you not to fork this repo because forks are ranked lower than non-forks.)
   >
   > In some sense, we are doing a "dual problem" to data mining by getting these stars.
   > Recruiters are using data mining to find out who the best people to recruit are,
   > and we are hacking their data mining algorithms by making those algorithms select you instead of someone else.
   >
   > If you're interested in exploring this idea further, here's a python tutorial for extracting GitHub's social graph: <https://www.oreilly.com/library/view/mining-the-social/9781449368180/ch07.html> ; if you're interested in learning more about how recruiters use github profiles, read this Hacker News post: <https://news.ycombinator.com/item?id=19413348>.

5. Submit the url of your repo to sakai.

   Each part is worth 2 points, for 12 points overall.
