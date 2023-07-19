#!/usr/bin/env python3

import aws_cdk as cdk

from stack.cdk_stack import WorkshopStack

app = cdk.App()
WorkshopStack(app, "genai-text-to-sql-workshop")
app.synth()
