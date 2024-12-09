# Similarities
Simple app for similar images search.
Supported extensions `jpg`, `jpeg`, `png`.

## Installation

### Requirements
- [Docker](https://docs.docker.com/engine/install/) and [Docker compose](https://docs.docker.com/compose/install/) installed in the system

### Installation

Get the code from Github:
```
git clone https://github.com/stan7123/similarities.git
```

### Running

#### Configure
Create file named `.env` based on `default.env` and adjust variables if needed (should work fine without adjustments).

#### Run
```
docker compose up --build
```
Or to run in the background:
```
docker compose up -d --build
```

Open http://localhost/healthcheck to confirm the service is running (should show "ok" status). 

## How to use

### Upload image

Images can be submitted by sending request with `multipart/form-data` content type and file in `image` field to `http://localhost/upload`.

An exemplary **curl** call: `curl -v -F image=@/path/to/image.png http://localhost/upload`

In a response to this call, there is an id of the image returned by which, we can download it or find similar images.

Note: Image size is currently limited to 50MB. It can be adjusted on the proxy.

### Download image

To download the image call: http://localhost/download/[image_id] . There will be redirection to the image direct URL.

An exemplary **curl** call: `curl -v http://localhost/download/99f557a0-3f00-4715-bb58-d74013ef541f`

### Find similar images

To find similar images in the service use: `http://localhost/similar/[image_id]/[search_type]` where:

- image_id - id returned during upload
- search_type - different types of search, available options are: **colors**, **objects**, **texture**

You can use **limit** and **max_distance** optional query params to refine your search. **limit** is **10** by default. **max_distance** has no default value. 

An exemplary **curl** calls: 
- `curl http://localhost/similar/99f557a0-3f00-4715-bb58-d74013ef541f/colors`
- `curl http://localhost/similar/99f557a0-3f00-4715-bb58-d74013ef541f/colors?limit=5`
- `curl http://localhost/similar/99f557a0-3f00-4715-bb58-d74013ef541f/texture?max_distance=1.5`


| Search type   | When to use          |
|----------|:------------|
| colors | Comparing images based on dominant colors (e.g., landscapes, fashion, art).  |
| objects | Analyzing shapes and structures (e.g., people, objects, architecture).      |
| texture | Identifying texture patterns (e.g., fabrics, wood, medical images). |

## API docs

API docs can be found at: `http://localhost/docs`


## Running tests

First you need to prepare test database:
```
docker compose exec db bash
psql -U db
CREATE DATABASE test_db;
\c test_db
CREATE EXTENSION vector;
```

Then you can run tests

```
docker compose run backend pytest
```


## Design decisions
- Using different histogram types for images and storing them as vectors. This is an efficient way for storing and search complexity
- Using postgres with pgvector extension because Mysql has no indexes on vector columns.
- Storing uploaded files in directories next to the code for simplicity. Usually some kind of storage service like S3 should be used.
- Using rq as simple and lightweight worker for background tasks
- All the components can easily scale (API, background workers, storage). This design should scale easily to millions of images. First bottleneck will be the database and when going further some other DB which can scale horizontally and has vector search support can be used e.g. MongoDB, Cassandra and potentially other specialized in vectors.   
- Image extensions(formats) are limited to: jpg, jpeg, png. The app can probably process many other formats - can be researched an extended.
- There was rather little effort put into domain topics like histogram's parameters tuning. 
- Background task for histogram calculation is retried 10 times with exponential backoff in case of error. After that, submitted images can be ignored or a periodical task (not implemented) might try to schedule them again for processing.  

## Things to improve for production setup
- Components might require replacement when run in the cloud for scaling and reliability: S3/Cloud storage for storing images, SQS or similar as queue instead of Redis etc.
- CI/CD workflows
- Introduce throttling to secure against API overload/abuse

## Possible optimizations
- Calculate image's hash and check if the image was already uploaded before. It would reduce computations and storage in case same images are uploaded multiple times.
- Maybe histogram calculations can be run using GPU
