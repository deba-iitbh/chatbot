import scrapy
import os
import html2text
from urllib.parse import urljoin


class IitbhilaispiderSpider(scrapy.Spider):
    name = "iitbhilaispider"
    allowed_domains = ["iitbhilai.ac.in"]
    start_urls = ["https://www.iitbhilai.ac.in", "https://polaris.iitbhilai.ac.in"]

    def __init__(self, *args, **kwargs):
        super(IitbhilaispiderSpider, self).__init__(*args, **kwargs)

        if not os.path.exists("../output"):
            os.makedirs("../output")

        self.converter = html2text.HTML2Text()
        self.converter.ignore_links = True
        self.converter.ignore_images = True
        self.converter.ignore_tables = True
        self.converter.body_width = 0
        self.visited_url = set()

    def parse(self, response):
        # Handle dead ends by checking if the response is valid
        if response.status != 200 or response.headers.get("content-type") in [
            b"application/pdf",
            b"application/zip",
            b"application/x-zip-compressed",
            b"application/docx",
            b"application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        ]:
            return

        # Check if URL has been visited before
        if response.url in self.visited_url:
            return

        try:
            text = self.converter.handle(response.body.decode("utf-8"))
            text = text.strip()

            url = response.url.strip("/")
            filename = f"../output/{hash(url)}.txt"
            with open(filename, "w") as f:
                f.write(text)

        except:
            print("Unable to parse URL:", response.url)

        try:
            # Extract links from the response
            links = response.css("a::attr(href)").getall()
        except:
            return

        for link in links:
            if link.startswith("/"):
                url = urljoin(response.url, link)
            elif link.startswith("https://www.iitbhilai.ac.in") or link.startswith(
                "https://polaris.iitbhilai.ac.in"
            ):
                url = link
            else:
                continue

            yield scrapy.Request(url, callback=self.parse)
