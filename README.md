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

The five test emails include a quoted urgent request that has already been resolved and a blocked submission without the word “urgent.” Each result reports the actual output, expected label, label validity, and whether it matches. The expected labels define the test; the script obtains predictions from the model.

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
python paw_email_urgency.py run ./urgency.paw
```

[Try the playground](https://programasweights.com/playground) · [Complete walkthrough](https://weightspaceprogramming.tech/build/reusable-neural-function/) · [SDK reference](https://programasweights.readthedocs.io/en/latest/api-reference/python-sdk/) · [Local compiler](https://github.com/programasweights/compiler) · [PAW paper](https://arxiv.org/abs/2607.02512)

## 3. Compose urgency and routing

Compile a second function that returns `ACADEMIC`, `OPERATIONS`, or `OTHER`. The Python application combines its destination with the urgency label to select a queue.

```sh
python paw_email_urgency.py compile --kind router --output router-program.json
python paw_email_urgency.py compose urgency-program.json router-program.json \
  --input "The release is waiting on your approval. Please unblock it now."
```

The script checks each label before using it. The neural functions make the semantic judgments; ordinary Python maps their outputs to destination and priority fields. It prints the decision without sending an email or changing a queue.

Change the urgency policy to require an explicit deadline today, then compare the new function with the original:

```sh
python paw_email_urgency.py compile --policy deadline --output deadline-program.json
python paw_email_urgency.py run deadline-program.json
```

Compare compiled execution with prompting the unadapted Qwen3-0.6B interpreter on the same test inputs:

```sh
python paw_email_urgency.py prompt
```

The prompt command supplies the policy on each call. Record the actual outputs and timings from both commands to compare them.

## Contributing an example

Include the source, installation commands, representative inputs, expected behavior, and the command that checks the result. Link the original paper and implementation. For model-generated outputs, report the model or compiler version and keep predictions separate from expected answers.
