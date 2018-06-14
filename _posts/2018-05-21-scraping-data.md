---
layout: post
title: Bears, Beets, NLP - Part 1
subtitle: Scraping The Office Script With Scrapy and BeautifulSoup
tags: [project, web-scraping]
---


The Office offers endless entertainment. Not only is it infinitely rewatchable, but it turns out that it lends itself to data science projects as well. I've been working on a project analyzing the the script of The Office and will be presenting the results as a series of blog posts here. My goal is to build a classifier to predict which character from The Office said (or would say) a given line. I'm experimenting by building a series of different classifiers. I'll start with a baseline model, and over the next couple posts will see if I can beat its performance by engineering new features or using different models.

All the relevant project code can be found in [the GitHub repository](https://github.com/conleyst/bears-beets-nlp) and all the current blog posts in the series are:

- [Part 1: Scraping The Office Script with Scrapy and BeautifulSoup](https://conleyst.github.io/2018-05-21-scraping-data/)

- [Part 2: Cleaning the Data and Exploratory Data Analysis](https://conleyst.github.io/2018-05-22-clean-eda/)

- [Part 3: Bag-Of-Words, Random Forest, TFIDF, and Logistic Regression](https://conleyst.github.io/2018-05-23-random-forest-log-reg/)

- [Part 4: Glove Embeddings and a Convolutional Neural Net for Text Classification](https://conleyst.github.io/2018-05-26-convnet/)

---

### Part 1: Scraping the Data Using Scrapy and Beautiful Soup

Getting the actual script of The Office is the first hurdle that needs to be cleared. To my knowledge, there isn't a single dataset containing the full script, but there is a nice website [OfficeQuotes](http://officequotes.net/) that contains the full script broken up over several pages. I scraped the data from the site, and there are two main tools that I used for this were:

1. [Scrapy](https://docs.scrapy.org/en/latest/)

    Scrapy is a Python package designed to allow us to build a web-crawler/web-scraper. The objects that Scrapy makes are called *Spiders*. A Scrapy Spider can follow links (web-crawl) and extract data from the HTML of the pages that it visits (web-scrape). I used Scrapy for its web-crawling capabilities, but for the actual web-scraping I used BeautifulSoup.

2. [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)

    BeautifulSoup is a Python package that is very useful for parsing HTML. Although Scrapy can also parse HTML, BeautifulSoup is created to do this one thing, and I find that it does it very well. It's a nice complement to Scrapy.

Let's run through how I scraped the lines off of the website. If you aren't familiar with HTML, you don't need to know much, but you should have some idea of what a tag or element is. If you don't, spend a few minutes clicking around this website [here](https://www.w3schools.com/html/html_intro.asp).

**Step 0: Install Scrapy and BeautifulSoup**

This one goes without saying, but you need both Scrapy and BeautifulSoup. Click the links above to check out the documentation and to see how to install them on your system. If you want the exact version I used, check out the `environment.yml` file that can be found in the GitHub repository for this project. If you have Anaconda, you also have the option of installing the environment from that file. Instructions for doing that are [here](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file).

**Step 1: Set-Up the file structure**

Scrapy expects certain directories and scripts to be in certain places. Luckily, it does all the work of putting them there itself. To set-up the project, navigate in the shell to wherever you want the crawler source code to be and use the command,

```python
scrapy startproject office_crawler
```

This will set-up a project called `office_crawler`.

Now we'll create the file that we're going to write the actual script we'll use to control our Spider. Navigate to `office_crawler/office_crawler/spiders/` and create a file `office_spider.py`, and put in the lines,

```python
import scrapy
from bs4 import BeautifulSoup


class OfficeSpider(scrapy.Spider):
    name = 'office_crawler'
    allowed_domains = ['officequotes.net']
    start_urls = [
        'http://officequotes.net/index.php'
    ]
```

We're going to add more to this, but all we've done here is declare that we want to create an object called an `OfficeSpider`, which we've named `office_crawler`. We've also specified that it can't leave the website we're interested in and we gave it a start point.

We still haven't said a thing about which links to follow or what to do when it goes to those links. We'll do this by adding two methods telling Scrapy how to parse the HTML. The first will tell it how to parse the HTML to find links and then travel to those links, and the second will tell it what to do when it follows those links.

**Step 2: Tell Scrapy which links you want it to follow**

Now we want to parse the HTML of the starting point we declared to extract links for our Spider to follow.

> There is *no* foolproof way of parsing HTML. Nearly all website have slightly differing HTML, many with very ugly, nested tags and elements (this is called tag soup, and is where BeautifulSoup got its name). You will need to manually inspect the HTML and use trial-and-error to figure out how to extract the content you want. The [Scrapy shell](https://doc.scrapy.org/en/latest/topics/shell.html) is a good place to experiment.

If you haven't looked at the HTML of the website yet, spend a few minutes doing so and try and determine how you would get the links -- notice they're all in the menu bar on the left.

I used the fact that the elements we want have the tag `<a>` and an `href` attribute. We can take advantage of Scrapy's ability to parse XPath commands to get the links,

```python
response.xpath('//a[contains(@href, "no")]/@href').extract()
```

Let's add this as a parse method,

```python
import scrapy
from bs4 import BeautifulSoup


class OfficeSpider(scrapy.Spider):
    name = 'office_crawler'
    allowed_domains = ['officequotes.net']
    start_urls = [
        'http://officequotes.net/index.php'
    ]

    # start at home page and retrieve links
    def parse(self, response):
        links = response.xpath('//a[contains(@href, "no")]/@href').extract()
        for link in links:
            url = "http://officequotes.net/" + link
            yield scrapy.Request(url, callback=self.parse_quotes)
```

A few notes:

1. The `parse` method is always called first when we start our Spider, so it should be the one extracting the links.
2. The `response` variable refers to the HTML returned when Scrapy goes to the website listed in our `start_urls` list.
3. Notice I also referred to a `parse_quotes` method that we haven't written yet. This is the method that will extract the quotes from the links we extracted.
4. I used `yield` instead of `return`. Return stops the script, but we want our Spider to keep going until it runs out of links. Using `yield` does this.

**Step 3: Tell Scrapy how to extract the lines from the links**

Again, if you haven't looked at the HTML of one of the pages containing the quotes, do so. Think about how you might isolate just the quotes from the page.

Once our Spider gets to a page, we want it identify just the quotes. All the quotes are boxed, and in the HTML, it means they're in their own elements, with their own `<div>` tag. For example,

```html
<div class="quote">
<b>Stanley:</b> I don't want to stay until seven again this year.<br/>
<b>Pam:</b> I don't really have any control over that Stanley.<br/>
</div>
```

We can use the fact that all quotes are of the `quote` class and extract them with,

```python
soup = BeautifulSoup(response.text, 'lxml')
all_quotes = soup.find_all('div', class_='quote')
```

This gives us a list of the elements, but we still need to parse them.

> When scraping a webpage, it can be beneficial to only do the minimum processing necessary to get what you want. It's easier to clean data locally than it is to clean it as you scrape it. Too much processing can introduce errors. Although Scrapy can handle errors and will continue scraping, it will skip that quote/page, and it's up to you to notice if something is missing.

We can pull out a list of the strings from an element using the `strings` method,

```python
['\n',
 'Stanley:',
 " I don't want to stay until seven again this year.",
 '\n',
 'Pam:',
 " I don't really have any control over that Stanley.",
 '\n']
```

Scrapy also wants us to have a standard way to return what we extract, and this means creating an Item. Think of an Item like a dictionary. Ideally, we want a new dictionary for every line with one entry for the character speaking and one for the line they said. Items are nice, because if for some reason we make a mistake, that will just create an empty dictionary and the script will keep going. The issue for us is that not every element follows the pattern character-line like in the example above. For example,

```html
```{html}
<div class="quote">
<b><u>Deleted Scene 14</u></b>
<div class="spacer">&amp;nbsp</div>
<b>Angela:</b> You behaved very badly tonight.<br/>
<b>Kelly:</b> Sorry?<br/>
</div>
```

produces the following when we call the `strings` method,

```python
['\n',
 'Deleted Scene 14',
 '\n',
 '&nbsp',
 '\n',
 'Angela:',
 ' You behaved very badly tonight.',
 '\n',
 'Kelly:',
 ' Sorry?',
 '\n']
```
Rather than try and parse all these exceptions on the fly, we'll create an Item with two entries. The first will be the list output of the `strings` method (minus an new line characters) and the second will be the URL we obtained the list from. The URL will allow us to later extract the season and episode.

Go to `office_crawler/office_crawler/items.py` and add the lines,

```python
import scrapy

class LineItem(scrapy.Item):
    conversation = scrapy.Field()
    url = scrapy.Field()
```

And then we create the second parse method, `parse_quotes`, that returns these items to us,

```python
import scrapy
from office_crawler.items import LineItem
from bs4 import BeautifulSoup


class OfficeSpider(scrapy.Spider):
    name = 'office_crawler'
    allowed_domains = ['officequotes.net']
    start_urls = [
        'http://officequotes.net/index.php'
    ]

    # start at home page and retrieve links, parse linked pages with parse_quotes
    def parse(self, response):
        links = response.xpath('//a[contains(@href, "no")]/@href').extract()
        for link in links:
            url = "http://officequotes.net/" + link
            yield scrapy.Request(url, callback=self.parse_quotes)

    # define what to output at each page
    def parse_quotes(self, response):
        soup = BeautifulSoup(response.text, 'lxml')
        all_quotes = soup.find_all('div', class_='quote')  # retrieve all div elements with tag 'quote'
        current_url = response.request.url
        items = []
        for quote in all_quotes:
            text = list(quote.strings)
            text = list(filter(lambda s: s != '\n', text))
            item = LineItem(
                conversation=text,
                url=current_url
            )
            items.append(item)
        return items

```

A few notes:

1. Notice we added a new import statement so we can use the item we declared.
2. We use `return` in the second method. Once we have all the quotes from the link, we're done with that page and want Scrapy to leave it.

**Step 4: Run the Spider**

Running the Spider we created is easy, but before we do, go into `settings.py` and add the line `DOWNLOAD_DELAY=1`. This will make our Spider limit the number of requests it send the web server. We don't want to overload the server with too many requests at once. This benefits the person hosting, since they're paying to host and we could crash the server or slow down others trying to use it, and it benefits us, since we otherwise could be blocked for sending too many requests.

Run the Spider by going into `office_crawler/` and using the command,

```
scrapy crawl office_crawler -o lines.json
```

This will output a JSON file with the scraped data that can be cleaned at our leisure.

In the next post, I'll show how I cleaned the data and we'll do some EDA.
