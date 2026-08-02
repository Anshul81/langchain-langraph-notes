# Redis Interview Guide (Based on the CodeScout / RapidX Project)

> A beginner-friendly, interview-ready walkthrough of **how Redis is actually used in this project**, plus the **general Redis concepts** interviewers love to ask.
>
> How to use this file: Read Part 1 to understand Redis basics, Part 2 to confidently talk about *your* project, and Part 3–4 to handle broader questions. Part 5 is a quick-revision cheat sheet.

---

## Part 1 — Redis in 5 minutes (the basics)

**What is Redis?**
Redis (REmote DIctionary Server) is an **in-memory, key-value data store**. Because it keeps data in RAM (not on disk like a normal database), reads/writes are extremely fast — typically sub-millisecond.

**Why do we use it?**
The most common reason (and *our* reason in this project) is **caching**: storing the result of an expensive operation (a database query, an API call) so the next time we need it, we read it from Redis instantly instead of redoing the expensive work.

**A simple mental model**
Think of Redis as a giant Python dictionary that:
- Lives in memory (fast),
- Is shared across all your servers/pods (centralized),
- Can automatically delete entries after some time (TTL / expiry).

**Key terms you must know**
| Term | Meaning |
|------|---------|
| **Key** | The unique name you store data under, e.g. `patterns:java`. |
| **Value** | The data stored (string, hash, list, etc.). |
| **TTL (Time To Live)** | How long a key lives before Redis auto-deletes it. |
| **Cache hit** | The data you wanted was already in Redis. |
| **Cache miss** | Not in Redis → you fetch from the real source (DB/API). |
| **Eviction** | Redis removing keys when memory is full. |
| **Invalidation** | *You* deliberately deleting/refreshing a cached key because the underlying data changed. |

---

## Part 2 — How Redis is used in THIS project

This project is a microservices system (parser, repository-analyser, snippet-analyser, insights-generator, and an IAM/auth layer). Redis is used in **three distinct ways**. Being able to describe these three is the core of your interview story.

### Use case A — Caching AI "prompt templates" per project

**The problem:** Each project has many prompt templates stored in a Postgres database. Services fetch these prompts constantly while analysing code. Hitting Postgres every single time is slow and wasteful because prompts rarely change.

**The solution:** Cache prompts in Redis so lookups are instant.

**Data structure used: Redis Hash.** A hash is like a "dictionary inside a value" — one key holds multiple fields. This is perfect because a prompt has several attributes (`prompt_id`, `text`, etc.).

**Writing to the cache** (repository-analyser service) — note three important techniques: a **pipeline** for bulk writes, `hset` to store a hash, and `expire` to set a TTL:

```python
def add_prompts_of_project(self, project_id, prompt_catalogs):
    pipeline = redis_connection.pipeline()           # batch many commands into ONE round trip
    for prompt_catalog in prompt_catalogs:
        hash_name = f"{prompt_catalog.type}_{prompt_catalog.attribute}_{project_id}"
        hash_mapping = {"prompt_id": prompt_catalog.id, "text": prompt_catalog.text}
        pipeline.hset(hash_name.lower(), mapping=hash_mapping)  # store as a hash
        pipeline.expire(hash_name, PROMPT_CACHE_TTL_IN_SEC)     # auto-expire (TTL)
    pipeline.execute()                                # send everything at once
```

**Reading from the cache** (snippet-analyser service) — this is the classic **Cache-Aside** pattern:

```python
def get_prompt(self, project_id, provider, tech, type, attribute, metadata):
    try:
        redis_key = f"{type}_{attribute}_{project_id}"
        redis_value = di[RedisService].get_prompt(redis_key)   # 1. Try Redis first
        if redis_value:
            return int(redis_value["prompt_id"]), redis_value["text"]   # CACHE HIT
    except Exception as err:
        _LOGGER.warning(f"Error while getting cached value - {redis_key}")

    # 2. CACHE MISS → fall back to the database
    prompt = di[ProjectPromptCatalogRepository].get_prompt_template(...)
    ...
```

**The golden design rule in this code (great to quote in an interview):**
> "The service should **not be dependent on Redis. It is just for performance improvement.**"

That comment appears in the code. It means Redis is treated as an **optional accelerator**, not a source of truth. If Redis is down, the code catches the exception and simply reads from Postgres. This is called **graceful degradation** and interviewers love hearing it.

**Cache invalidation** — when prompts change, stale entries are deleted using pattern matching:

```python
def delete_prompts_by_project(self, project_id):
    for name in redis_connection.scan_iter(f"*_{project_id}"):  # find all keys for this project
        redis_connection.delete(name)                            # delete them
```

> **Interview note on `scan_iter` vs `KEYS`:** `scan_iter` uses the `SCAN` command, which walks the keyspace in small batches. This is **production-safe**. The `KEYS` command does the same search but **blocks the entire Redis server** while it runs — dangerous on large datasets. Knowing this difference is a strong signal of practical Redis experience.

---

### Use case B — Caching parsed "language patterns"

**The problem:** The parser looks up coding patterns for a language (Java, C++, etc.). These come from a database and change rarely.

**The solution:** Cache the whole list as a JSON string under a simple key.

**Data structure used: String** (storing serialized JSON).

```python
def get_language_patterns(self, language):
    key = f"patterns:{language}"
    data = redis_connection.get(key)           # GET a string
    return json.loads(data) if data else None  # deserialize JSON

def set_language_patterns(self, language, patterns):
    key = f"patterns:{language}"
    redis_connection.set(key, json.dumps(patterns))   # SET a JSON string

def refresh_redis_cache(self, language):
    redis_connection.delete(f"patterns:{language}")    # invalidate on change
```

**Talking point — Hash vs String, why two approaches?**
- Use case A stores **structured fields** → a **Hash** lets you read individual fields without deserializing everything.
- Use case B stores a **whole list/blob** → a **String of JSON** is simpler when you always read the entire thing at once.

Being able to justify *why each data structure was chosen* is exactly the kind of reasoning interviewers reward.

---

### Use case C — IAM authentication / session caching (the advanced one)

This is the most sophisticated Redis usage in the project (`codebits/iam/`). It caches **user identity and permissions** so that every API request doesn't have to call the IAM (auth) microservice. It uses several advanced patterns — if you can explain even two of these, you'll stand out.

**1. Write-through caching + TTL via `SETEX`**
After fetching a user from the IAM service, we immediately store it in Redis with an expiry:

```python
self._redis.setex(cache_key, extended_ttl, json.dumps(cache_entry))
```
`SETEX` = "set with expiry" in one atomic command.

**2. Very short TTL for security**
Auth data uses a **1–5 second TTL** (the code even warns if TTL > 5s). Why so short? If an admin **revokes a user's permission**, you don't want a stale "you're allowed" answer cached for long. This is the classic **security vs. performance trade-off** — a fantastic thing to discuss.

**3. Stale-While-Revalidate (graceful fallback)**
Each entry stores a `fresh_until` timestamp and is kept a bit longer than its "fresh" window (a *grace period*). If the fresh copy expired but the IAM service is momentarily slow/down, the code can still serve the *slightly stale* copy instead of failing the request:

```python
fresh = now < cache_entry["fresh_until"]
return {"data": cache_entry["data"], "fresh": fresh}
```

**4. Distributed lock → "cache stampede" protection**
Problem: when a popular key expires, hundreds of requests may all miss the cache at once and hammer the IAM service simultaneously (a "stampede" / "thundering herd").
Solution: the **first** request grabs a lock using `SET key value NX EX` (set *only if Not eXists*, with an *EXpiry*), fetches the data, and fills the cache. Everyone else waits or uses stale data.

```python
acquired = self._redis.set(lock_key, "1", nx=True, ex=_LOCK_TIMEOUT)  # atomic lock
```
Other requests use **jittered backoff** (randomized small sleeps) while waiting, to avoid all retrying at the exact same instant.

**5. Negative caching (token revocation)**
When a token is revoked (logout / security incident), a "tombstone" key is written so the cached user data is ignored even if it still exists:

```python
revoked_key = f"{cache_key}:revoked"
self._redis.setex(revoked_key, _REVOKED_TOKEN_TTL, "1")  # remember "this is revoked" for 1 hour
```

**6. Double-checked locking**
After acquiring the lock, the code checks the cache *again* — another request may have already populated it while we waited. This avoids redundant work.

---

### Infrastructure — how Redis is deployed here

You can casually mention the ops side; it shows end-to-end understanding.

- **Managed Redis**, not self-hosted. On Azure it's **Azure Cache for Redis** (Premium tier, defined in Terraform / Bicep); AWS deployments use **ElastiCache** modules.
- **Security:** TLS/SSL enforced on **port 6380**, non-SSL port disabled, and **private endpoints** (no public network access) so only the app's virtual network can reach it.
- **Configuration via environment variables** (from Kubernetes ConfigMaps/Secrets), never hard-coded:

```python
redis_connection = Redis(
    host=os.environ.get("REDIS_HOST"),
    port=int(os.environ.get("REDIS_PORT")),   # 6380
    password=os.environ.get("REDIS_PASSWORD"),
    ssl=True,
    decode_responses=True,   # return Python str instead of raw bytes
)
```

- **Shared cache across pods:** because Redis is a central server, all replicas of a service share the same cache — a request handled by pod A benefits from data cached by pod B.

---

## Part 3 — Core Redis concepts (general interview knowledge)

### Data structures (know at least these 5)
| Type | Description | Example use here / typical use |
|------|-------------|-------------------------------|
| **String** | Simplest; text, numbers, or JSON blobs. | Our `patterns:{language}` JSON cache; counters. |
| **Hash** | Field→value map under one key. | Our prompt cache (`prompt_id`, `text`). |
| **List** | Ordered, allows push/pop from ends. | Queues, recent-activity feeds. |
| **Set** | Unordered unique items. | Tags, unique visitor tracking. |
| **Sorted Set (ZSet)** | Set where each member has a score, kept sorted. | Leaderboards, rate limiting, priority queues. |

Also good to name-drop: **Streams** (event logs), **Bitmaps**, **HyperLogLog** (approximate unique counts), **Geospatial**.

### Common caching patterns
- **Cache-Aside (Lazy Loading):** App checks cache; on miss, loads from DB and writes to cache. *(Used in this project for prompts.)*
- **Write-Through:** Write to cache and DB together, so cache is always fresh. *(Used in the IAM layer.)*
- **Write-Behind (Write-Back):** Write to cache first, flush to DB asynchronously later. Fast but riskier.
- **Read-Through:** The cache library itself loads from DB on a miss (app only talks to cache).

### TTL & eviction
- **TTL / Expiry:** `EXPIRE key seconds`, or set at write time with `SETEX` / `SET key val EX 60`.
- **Eviction policies** (what Redis does when memory is full): `noeviction`, `allkeys-lru`, `allkeys-lfu`, `volatile-lru`, `volatile-ttl`, etc. **LRU** = evict Least Recently Used; **LFU** = Least Frequently Used.

### Persistence (Redis can survive restarts)
- **RDB:** point-in-time snapshots to disk. Compact, fast restart, but can lose recent writes.
- **AOF (Append Only File):** logs every write; more durable, larger files.
- Many production setups use **both**. (For pure caching, persistence is often less critical.)

### Atomicity & single-threaded model
- Redis command processing is effectively **single-threaded**, so each command is **atomic** — no partial/interleaved execution. This is *why* `SET ... NX` works as a reliable lock.
- **Pipelines** batch multiple commands in one network round-trip (fewer round-trips = faster). *(Used in `add_prompts_of_project`.)* Pipelines are **not** transactions by themselves.
- **Transactions:** `MULTI` / `EXEC` queue commands and run them together; `WATCH` adds optimistic locking.
- **Lua scripts / `EVAL`:** run custom logic atomically on the server.

### Scaling & availability
- **Replication:** primary–replica copies for read scaling and failover.
- **Redis Sentinel:** monitors and does automatic failover for high availability.
- **Redis Cluster:** shards data across nodes using **hash slots** (16384 slots) for horizontal scaling.

### Pub/Sub & beyond caching
- **Pub/Sub:** publish/subscribe messaging channels.
- **Redis as a message queue / rate limiter / distributed lock / leaderboard** are all common non-cache uses.

---

## Part 4 — Likely interview Q&A (with project-grounded answers)

**Q: Where and why did you use Redis in your project?**
A: In a microservices code-analysis platform, primarily as a **cache**. Three places: (1) caching per-project AI prompt templates that otherwise came from Postgres, (2) caching language patterns as JSON, and (3) caching user identity/permissions in the IAM layer to avoid calling the auth service on every request. The goal was **lower latency and reduced load** on Postgres and the IAM service.

**Q: What data structures did you use and why?**
A: **Hashes** for prompts (structured fields like `prompt_id` and `text` under one key), and **Strings** holding JSON for language-pattern lists (always read as a whole). Choice depends on access pattern.

**Q: What happens if Redis goes down?**
A: The services are designed so Redis is **optional** — the code literally comments "service should not be dependent on Redis." All Redis calls are wrapped in try/except; on failure we log a warning and fall back to Postgres. This is **graceful degradation**.

**Q: How do you keep the cache from serving stale data?**
A: Two mechanisms — **TTL** (keys auto-expire, e.g. prompts via `EXPIRE`, IAM data with 1–5s `SETEX`), and **explicit invalidation** (deleting keys when the source changes, e.g. `scan_iter("*_{project_id}")` then `delete`).

**Q: What is a cache stampede and how did you handle it?**
A: When a hot key expires, many requests miss simultaneously and overload the backend. In the IAM layer we use a **distributed lock** (`SET NX EX`) so only one request refreshes the value while others wait (with jittered backoff) or serve stale data. We also use **double-checked locking** after acquiring the lock.

**Q: Why did you use a very short TTL for auth data?**
A: Security. Cached permissions could otherwise stay valid after being revoked. A 1–5s TTL bounds that exposure window — a deliberate **security-vs-performance trade-off**, softened by a stale-while-revalidate grace period.

**Q: `KEYS` vs `SCAN`?**
A: `KEYS` blocks the whole server while scanning — unsafe in production. `SCAN` (`scan_iter` in redis-py) iterates in small batches without blocking. We used `scan_iter` for invalidation.

**Q: What's a Redis pipeline? Did you use it?**
A: A pipeline batches many commands into one network round-trip. Yes — when caching all prompts for a project, we pipeline the `hset`/`expire` calls and `execute()` once, instead of many separate round-trips.

**Q: How is Redis secured/deployed in your project?**
A: Managed service (Azure Cache for Redis / AWS ElastiCache), TLS on port 6380, non-SSL disabled, private endpoints so it's not publicly reachable, and credentials injected via environment variables from Kubernetes config/secrets.

**Q: Redis vs a normal database?**
A: Redis is in-memory (fast, volatile-by-default, simpler data model); a relational DB is on-disk, durable, and supports complex queries/joins. We use Redis *in front of* Postgres, not instead of it.

**Q: `SET key val NX EX 10` — what does each part do?**
A: `NX` = set only if the key does **N**ot e**X**ist; `EX 10` = expire after 10 seconds. Together they make a safe, self-releasing lock.

---

## Part 5 — Quick-revision cheat sheet

**Commands seen / relevant in this project**
```text
SET key value            # store a string
GET key                  # read a string
SETEX key ttl value      # set with expiry (IAM cache)
SET key val NX EX 10     # atomic lock: set-if-absent + expire
HSET key field value     # set a field in a hash (prompt cache)
HGETALL key              # read all fields of a hash
EXPIRE key seconds       # add/refresh a TTL
TTL key                  # seconds left before expiry
DEL key                  # delete a key (invalidation)
EXISTS key               # does key exist? (revoked-token check)
SCAN / scan_iter         # safe keyspace iteration (invalidation)
KEYS pattern             # AVOID in prod — blocks server
MULTI / EXEC             # transaction
EVAL                     # run Lua script atomically
```

**One-liner summary to memorize:**
> "In this project Redis is an **in-memory cache in front of Postgres and the IAM service**. I used **hashes** and **JSON strings** with **TTLs** and **explicit invalidation** (cache-aside), plus advanced patterns in the auth layer — **write-through**, **stale-while-revalidate**, **distributed locks (SET NX EX) for stampede protection**, and **negative caching for revoked tokens** — all designed so the app **degrades gracefully if Redis is unavailable**."

**Three trade-offs you can always discuss:**
1. **Speed vs. staleness** → solved with TTL + invalidation.
2. **Security vs. performance** → short TTL for auth data.
3. **Availability vs. correctness** → stale-while-revalidate and graceful fallback to the DB.
