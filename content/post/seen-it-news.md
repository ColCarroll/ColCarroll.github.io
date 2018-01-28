+++
date = "2018-01-28T12:16:15-04:00"
tags = []
draft = false
title = "That was cool when the Mercury News did it 8 years ago"
highlight = true
math = false
summary = "A twitter bot that has read it all"
+++

In an effort to show how [Media Cloud](https://mediacloud.org/) can be used in simple open source 
projects, I [built a twitter bot](https://github.com/colcarroll/seen_it_news) this weekend. The 
premise is that the twitter bot [@NYT_first_said](https://twitter.com/NYT_first_said) tweets 
words whenever the New York Times uses them for the first time. My bot, 
[@seen_it_news](https://twitter.com/seen_it_news), searches a bunch of English language news 
sources (Washington Post, Wall St. Journal, USA Today, FOX News, among others) for uses of the word, 
and replies.

<blockquote class="twitter-tweet" data-lang="en"><p lang="en" dir="ltr">The Wall Street Journal was saying &quot;rewirement&quot; for 4 years before the New York Times did:<a href="https://t.co/O69rt0M3oh">https://t.co/O69rt0M3oh</a></p>&mdash; Hipster Media (@seen_it_news) <a href="https://twitter.com/seen_it_news/status/957010609572405248?ref_src=twsrc%5Etfw">January 26, 2018</a></blockquote>
<script async src="https://platform.twitter.com/widgets.js" charset="utf-8"></script>

It was a fun little project! I would love if the Media Cloud API could 
[look up sources by url](https://github.com/berkmancenter/mediacloud/issues/214) so this could
be entirely automated, but one cannot be too picky! [Tweepy](https://github.com/tweepy/tweepy) was
a joy to work with, as was [grift](https://github.com/kensho-technologies/grift), which was 
open-sourced at my previous company.

Anyways, give the bot a follow! My favorite discovery was that "metoo" was first used three years
ago by USA Today, as an answer 
[in a quiz about the original name of Yahoo](https://www.usatoday.com/story/money/columnist/strauss/2015/08/17/strauss-small-business-quiz/31754211/)!
