# Problem statement

This section defines what the problem is we want to tackle, the broader context.

## Work context

- I work in the integrations team at a B2B SaaS business.
- I integrate with lots of different systems (currently ~20)
- While systems I integrate with are often the same across different customers, their data-formats aren't
- The data we get is most often tabular data, or data that can be transformed into tabular data (mostly CSV data)
- While we can't force this always, we try to follow the CSV reference when asking our customes to provide us with data ([CSV Reference](https://datatracker.ietf.org/doc/html/rfc4180))
- File sizes can get large but not insane. Large files are 1-10Gb
- We want to ingest all of this data into our well-defined API, but as said before, data is not consistent between customers.

## Actual problem statement

The problem is transforming tabular data of quasi random formats into a structured well-defined format that can be ingested by an API.

## Broader context

- Current solutions to this problem are costly
- are not integratable into your own environment
- are not built with expanding of a product in mind
- you cannot apply your own AI models easily to the problem since it is closed-source.
- it is desirable to allow customers (in a B2B business) to configure transformations themselves (with the help of AI tools)
- it must be workable by both very non-technical people (the target group) and computer scientists or other technical people (to be able to debug and develop)

## Stakeholders

People that will benefit from this:

- software developers looking to transform data into different formats
- software developers looking at this tool to take ideas from and implement into their own product
- companies looking for an almost out-of-the-box transformation solution
- Businesses that benefit from the products built on this tool

For businesses that have the following pattern (a baseline):

- Customers provide data via file uploads or shared drives
- Our team inspects files manually and writes Python scripts to parse and map them
- We use internal libraries to validate and transform data into our API format
- Debugging is slow and error-prone, especially when file formats change unexpectedly

## Desired End State

- A system where customers can define or review data mappings with AI assistance
- Transformations should be traceable, explainable, and editable by both technical and non-technical users
- AI should assist by proposing transformations, identifying anomalies, and validating consistency
