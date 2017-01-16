+++
# Date this page was created.
date = "2015-09-28"

# Project title.
title = "Cross Country Predictions"

# Project summary to display on homepage.
summary = "Using hundreds of thousands of historical cross country running results to make predictions about future meets. The page is updated more than weekly during the season."

# Optional image to display on homepage (relative to `static/img/` folder).
image_preview = "xc_pred.png"

# Optional image to display on project detail page (relative to `static/img/` folder).
image = ""

# Tags: can be used for filtering projects.
# Example: `tags = ["machine-learning", "deep-learning"]`
tags = ["javascript", "machine-learning", "running", "python"]

# Optional external URL for project (replaces project detail page).
external_link = ""

# Does the project detail page use math formatting?
math = false

+++

> ## *"There is not a publicly available source of cross country result data."*

**Short version:** tfrrs.org will send you terrifying letters if you try to analyze cross country data, and the USTFCCCA is ok with this.

**What you can do:** Communicate with people like they are humans. Especially if they share an interest in something you're passionate about. Also, [email me](http://localhost:8000/#/about) if you know of publicly available cross country results!

**Long version:** The [USTFCCCA](http://www.ustfccca.org/) ("U.S. Track & Field and Cross Country Coaches Association", who lost their ampersand-related nerve halfway through the naming process) has some sort of relationship with [tfrrs.org](https://www.tfrrs.org/) ( "Track & Field Results Reporting System"), providing them with cross country results. tfrrs.org does a fantastic job putting all these results into one place, and mostly in a consistent format. It is hard to find a popular organization that does not encourage people doing interesting things with their data: just in sports, the [NHL](http://www.nhl.com/stats/), [NBA](http://stats.nba.com/), [MLB](https://github.com/baseballhackday/data-and-resources/wiki/Resources-and-ideas), [NCAA football](http://developer.sportradar.com/files/indexFootball.html#ncaa-football-api), and [NCAA basketball](http://developer.sportradar.com/files/indexBasketball.html#daily-schedule45) have public apis encouraging non-commercial uses. The running app Strava also provides a [great looking api](https://strava.github.io/api/) for public use. I used the tfrrs.org results to make predictions about upcoming races, and they were as or more accurate than those published by more dedicated publishers of running news.

The project was open source, had a small static website to display some of the analysis, and I was getting an average of 1 visitor per day, who I can only assume was lost. I made no money off the project for any number of reasons:

*   No one but me was interested in it.
*   I already have a day job.
*   They weren't my results.
*   I was using [a very cool library from Microsoft](http://trueskill.org/) that prohibits commercial use.

In any case, the CEO of TFRRS sent me a cease and desist letter, which you can read at the bottom. The deleted tweet was a link to my site, and the text "Cross country analytics! \o/". I complied with all of his demands because I have other things to worry about, but made this page because I think this is a disgusting reaction to a fan.

As a followup, I emailed Tom Lewis, the Director of Media, Broadcasting and Analytics at the USTFCCCA asking if there is any public access to results.

His reply:

>   Subject: Re: Access to results?
>
>   Hi Colin,
>
>   No, there is not a publicly available source of cross country result data.
>
>   Tom
>
>   *---*
>
>   *Tom Lewis*Director of Media, Broadcasting, and Analytics

**Email from David Stelnik:**

> Subject: DirectAthletics/TFRRS **IMMEDIATE ACTION NEEDED**
>
> Mr. Carroll,
>
> It has come to our attention that you are scraping data off of our websites
> and actively and publicly soliciting it to commercial entities, in clear
> violation of our terms of use.
>
> We insist that you immediately take the following actions:
>
>   - Cease and desist from scraping our servers without our express,
>   written permission
>
>   - Cease and desist from soliciting and/or distributing our data,
>   including but not limited to any meet results, performances, performance
>   lists, rosters, teams or venues
>
>   - Remove the following projects from GitHub, as well as any other
>   publicly-available projects containing tools designed to scrape our sites
>   without our permission.
>     - https://github.com/ColCarroll/runpy
>     - https://github.com/ColCarroll/cross_country_analyzer/
>
>   - Tell us the names and contact information of any persons or companies
>   to which you distributed our data
>
>   - Delete all tweets [HERE](https://twitter.com/colindcarroll/status/788014277189828608)
>
> Your prompt attention to this matter is of the utmost importance.  If you
> have any questions regarding this matter, please reply to this email or
> call me at XXX-XXX-XXXX.
>
> Thank you very much,
>
> David
>
> Co-Founder
> DirectAthletics
