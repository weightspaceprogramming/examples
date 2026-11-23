# Weight-Space Programming Examples

Run a formal program compiled into transformer weights, then build reusable neural functions from English specifications.

These examples accompany the [Weight-Space Programming guide](https://weightspaceprogramming.tech/). Read [how the approaches compare](https://weightspaceprogramming.tech/learn/comparing-methods/) or browse the [papers, code, and models](https://weightspaceprogramming.tech/research/).

## 1. Compile a token counter with Tracr

[Tracr](https://github.com/google-deepmind/tracr) translates RASP programs into transformer weights. This example defines a program that counts the token `a`, compiles it, runs the transformer, and compares its outputs with both the RASP interpreter and the exact count. It runs on CPU without training or a hosted model.

Create a Python 3.12 environment:

```sh
python3.12 -m venv .venv-tracr
source .venv-tracr/bin/activate
python -m pip install -r requirements-tracr.txt
python tracr_count.py
```

For `abac`, the expected result is `[2, 2, 2, 2]`: each sequence position contains the total count. The script prints the actual transformer output alongside that expectation and exits with an error if they differ.

Check every sequence of length one through four over `a`, `b`, and `c`, then test length-eight boundaries:

```sh
python tracr_count.py --all-up-to 4
python tracr_count.py --sequence aaaaaaaa --sequence bbbbbbbb --sequence abababab
```

The compiled model supports one to eight tokens from that vocabulary. Change the RASP program to change the computation; changing the input sequence runs the same compiled transformer again.

[Walkthrough: compile an exact program](https://weightspaceprogramming.tech/build/compiling-exact-programs/) · [Tracr paper](https://arxiv.org/abs/2301.05062)

## 2. Compile and reuse an email-urgency function with PAW

[Program-as-Weights](https://programasweights.com/) turns a natural-language function specification into a reusable neural program. PAW specifically trains a compiler to generate LoRA adapters that edit the interpreter’s weights. The application calls the saved program on new emails.

The example defines this policy:

> Return URGENT for an imminent deadline or immediate blocker. Return ROUTINE for newsletters, completed work, and FYI updates.

Use a separate environment for the [PAW SDK](https://github.com/programasweights/programasweights-python):

```sh
python3.12 -m venv .venv-paw
source .venv-paw/bin/activate
python -m pip install -r requirements-paw.txt
python paw_email_urgency.py compile --output urgency-program.json
python paw_email_urgency.py run urgency-program.json
```

Compilation sends the specification to the PAW service. The SDK downloads the resulting program and its shared interpreter on first use, then executes inputs locally. The JSON record preserves the specification, program ID, and compiler snapshot.

The five test emails include a quoted urgent request that has already been resolved and a blocked submission without the word “urgent.” Each result reports the actual output, expected label, label validity, and whether it matches. The expected labels define the test; the script obtains predictions from the model. It prints a summary and exits with a nonzero status if any label is invalid or differs from the expectation. A custom input checks label validity without assigning an expected answer.

Run another email through the same program:

```sh
python paw_email_urgency.py run urgency-program.json \
  --input "The submission is ready, but the team cannot send it until you grant access."
```

Prepare assets and execute offline:

```sh
python paw_email_urgency.py prepare urgency-program.json
python paw_email_urgency.py run urgency-program.json --offline
```

Download the program as a portable file:

```sh
python paw_email_urgency.py download urgency-program.json --output urgency.paw
python paw_email_urgency.py run ./urgency.paw --policy original
```

When running the five test cases from a `.paw` file or bare program ID, pass `--policy original` or `--policy deadline`. Saved JSON records supply their policy automatically; `--policy` overrides it. The download command waits up to 120 seconds while the service prepares the artifact and preserves any existing output file.

[Try the playground](https://programasweights.com/playground) · [Complete walkthrough](https://weightspaceprogramming.tech/build/reusable-neural-function/) · [SDK reference](https://programasweights.readthedocs.io/en/latest/api-reference/python-sdk/) · [Local compiler](https://github.com/programasweights/compiler) · [PAW paper](https://arxiv.org/abs/2607.02512)

## 3. Compose urgency and routing

Compile a second function that returns `ACADEMIC`, `OPERATIONS`, or `OTHER`. The Python application combines its destination with the urgency label to select a queue.

```sh
python paw_email_urgency.py compile --kind router --output router-program.json
python paw_email_urgency.py compose urgency-program.json router-program.json \
  --input "The release is waiting on your approval. Please unblock it now."
```

The script checks each label before using it. The neural functions make the semantic judgments; ordinary Python maps their outputs to destination and priority fields. It prints the decision without sending an email or changing a queue.

Change the urgency policy to require an explicit deadline today, then compare the new function with the original. “Now,” “immediately,” and “ASAP” do not state a deadline today:

```sh
python paw_email_urgency.py compile --policy deadline --output deadline-program.json
python paw_email_urgency.py run deadline-program.json
```

Compare compiled execution with prompting the unadapted Qwen3-0.6B interpreter on the same test inputs:

```sh
python paw_email_urgency.py prompt
```

The prompt command supplies the policy on each call. Record the actual outputs and timings from both commands to compare them.

## Verified results

Recorded on November 22, 2026 (America/Toronto). The email tests used the [tested script](https://github.com/weightspaceprogramming/examples/blob/41fc6ee8bd980f23618dc0720924098931530b9a/paw_email_urgency.py). Hosted compilation used `paw-4b-qwen3-0.6b-20260407`; local inference used Qwen3-0.6B Q6_K, PAW SDK 0.4.14, and llama-cpp-python 0.3.36. Both compiled execution and the prompt baseline used greedy decoding with a 2,048-token context. The prompt baseline used the unadapted interpreter with thinking disabled.

| Test | Correct outputs |
| --- | --- |
| Compiled urgency | 5/5 |
| Compiled deadline policy | 8/8 (five emails + three additional checks) |
| Compiled routing | 9/9 |
| Compiled browser-tool reranking | 2/2 |
| Compiled website intent routing | 3/3 |
| Local compiler artifact, offline CPU inference | 5/5 |
| Prompt baseline, original policy | 5/5 |
| Prompt baseline, deadline policy | 4/5 |

### Original urgency policy

| Email | Expected | PAW output | Prompt output |
| --- | --- | --- | --- |
| Please approve the submission by 5 p.m. today so we can meet the deadline. | URGENT | URGENT | URGENT |
| Here is next month's seminar schedule for your information. | ROUTINE | ROUTINE | ROUTINE |
| The release is waiting on your approval. Please unblock it now. | URGENT | URGENT | URGENT |
| Forwarding the earlier ‘URGENT: signature needed’ email for the record. This was resolved yesterday. | ROUTINE | ROUTINE | ROUTINE |
| The submission is ready, but the team cannot send it until you grant access. | URGENT | URGENT | URGENT |

### Deadline-only policy

| Email | Expected | PAW output | Prompt output |
| --- | --- | --- | --- |
| Please approve the submission by 5 p.m. today so we can meet the deadline. | URGENT | URGENT | URGENT |
| Here is next month's seminar schedule for your information. | ROUTINE | ROUTINE | ROUTINE |
| The release is waiting on your approval. Please unblock it now. | ROUTINE | ROUTINE | URGENT |
| Forwarding the earlier ‘URGENT: signature needed’ email for the record. This was resolved yesterday. | ROUTINE | ROUTINE | ROUTINE |
| The submission is ready, but the team cannot send it until you grant access. | ROUTINE | ROUTINE | ROUTINE |

PAW returned ROUTINE for “Please unblock it now.” The prompt baseline returned URGENT. “Now” does not specify a deadline today, so the baseline scored 4/5 and the test command exited with status 1.

### Additional deadline checks

| Email | Expected | PAW output |
| --- | --- | --- |
| Please send your comments by noon today. | URGENT | URGENT |
| The build is broken. Fix it immediately. | ROUTINE | ROUTINE |
| Approve the budget by 9 a.m. tomorrow. | ROUTINE | ROUTINE |

### Routing outputs

| Email | Expected | PAW output |
| --- | --- | --- |
| The release is waiting on your approval. Please unblock it now. | OPERATIONS | OPERATIONS |
| Please approve the submission by 5 p.m. today so we can meet the deadline. | ACADEMIC | ACADEMIC |
| Here is next month's seminar schedule for your information. | ACADEMIC | ACADEMIC |
| The database service is unavailable. | OPERATIONS | OPERATIONS |
| Would you like to meet for dinner? | OTHER | OTHER |
| The submission is ready, but the team cannot send it until you grant access. | ACADEMIC | ACADEMIC |
| Can you teach the Monday class? | ACADEMIC | ACADEMIC |
| Our deployment is complete. | OPERATIONS | OPERATIONS |
| What is the weather forecast? | OTHER | OTHER |

Combining the original urgency function and router on the release-approval email produced:

```json
{
  "input": "The release is waiting on your approval. Please unblock it now.",
  "urgency": "URGENT",
  "destination": "operations",
  "queue": "priority"
}
```

### Additional functions

| Function | Input | Expected | PAW output |
| --- | --- | --- | --- |
| Browser-tool reranking | A client-side WebAssembly tool that runs entirely in your browser | exact_match | exact_match |
| Browser-tool reranking | A desktop-only application with a website describing its features | not_relevant | not_relevant |
| Website intent routing | I want to create a new function | playground | playground |
| Website intent routing | Where is the API reference? | docs | docs |
| Website intent routing | What is the weather? | none | none |

### Local compilation and offline execution

The local Standard compiler ran on an NVIDIA RTX 6000 Ada in 39.0 seconds, with 7.6 GiB peak GPU memory. Its saved artifact returned the same five outputs as the original-policy PAW column above during offline CPU inference: 5/5. That run used PAW SDK 0.4.14 and llama-cpp-python 0.3.34.

Saved JSON loading, .paw download and loading, and composition worked. Seven offline configurations ran with Python network connections blocked and recorded zero connection attempts. The deadline prompt baseline retained its one incorrect label in the offline run. All 11 validation and download regression tests passed.

Run the two comparisons:

```sh
python paw_email_urgency.py run urgency-program.json
python paw_email_urgency.py prompt
python paw_email_urgency.py run deadline-program.json
python paw_email_urgency.py prompt --policy deadline
```

[Download the full results, specifications, and program IDs](verified-results.json).

## Contributing an example

Include the source, installation commands, representative inputs, expected behavior, and the command that checks the result. Link the original paper and implementation. For model-generated outputs, report the model or compiler version and keep predictions separate from expected answers.

Check the script’s validation and download behavior without model calls:

```sh
python -m unittest test_paw_email_urgency.py
```
