# Google's Agent Development Kit (ADK) Research Summary

## Overview

The Agent Development Kit (ADK) is an open-source, model-agnostic, and deployment-agnostic framework developed by Google for building, evaluating, and deploying AI agents. The ADK is designed to make agent development feel more like traditional software development, simplifying the creation, deployment, and orchestration of agents for tasks ranging from simple to complex workflows.

## Core Components

### Agents

In ADK, an **Agent** is a self-contained execution unit designed to act autonomously to achieve specific goals. The foundation for all agents is the `BaseAgent` class, which can be extended in three main ways:

1. **LLM Agents (`LlmAgent`, `Agent`)**
   - Utilize Large Language Models (LLMs) as their core engine
   - Understand natural language, reason, plan, generate responses
   - Dynamically decide how to proceed or which tools to use
   - Ideal for flexible, language-centric tasks

2. **Workflow Agents**
   - Control the execution flow of other agents in predefined patterns
   - Types include:
     - **Sequential Agents**: Execute steps in a specified order
     - **Parallel Agents**: Execute multiple steps simultaneously
     - **Loop Agents**: Repeat execution of contained agents
   - Perfect for structured processes needing predictable execution

3. **Custom Agents**
   - Created by extending `BaseAgent` directly
   - Implement unique operational logic, specific control flows, or specialized integrations
   - Cater to highly tailored application requirements

### Tools

Tools in ADK represent specific capabilities provided to an AI agent, enabling it to perform actions beyond its core text generation and reasoning abilities.

Types of tools include:

1. **Function Tools**
   - User-defined tools tailored to specific application needs
   - Examples:
     - Functions/Methods: Standard synchronous functions or methods
     - Agents-as-Tools: Using another specialized agent as a tool
     - Long Running Function Tools: For asynchronous operations

2. **Built-in Tools**
   - Ready-to-use tools provided by the framework
   - Examples: Google Search, Code Execution, Retrieval-Augmented Generation (RAG)

3. **Third-Party Tools**
   - Integration with popular external libraries
   - Examples: LangChain Tools, CrewAI Tools

4. **Google Cloud Tools**
   - Integration with Google Cloud services

5. **MCP Tools**
   - Tools for the Model, Context, and Policy (MCP) framework

6. **OpenAPI Tools**
   - Tools generated from OpenAPI specifications

### Sessions & Memory

ADK provides structured ways to manage context through three key concepts:

1. **Session**
   - Represents a single, ongoing interaction between a user and an agent system
   - Contains the chronological sequence of messages and actions (Events)
   - Can hold temporary data (State) relevant only during this conversation

2. **State** (`session.state`)
   - Data stored within a specific Session
   - Used to manage information relevant only to the current, active conversation

3. **Memory**
   - Represents a store of information that might span multiple past sessions
   - Acts as a knowledge base the agent can search to recall information beyond the immediate conversation

### Deployment

ADK agents can be deployed to different environments:

1. **Agent Engine in Vertex AI**
   - A fully managed auto-scaling service on Google Cloud
   - Specifically designed for deploying, managing, and scaling AI agents

2. **Cloud Run**
   - A managed auto-scaling compute platform on Google Cloud
   - Enables running agents as container-based applications

3. **GKE (Google Kubernetes Engine)**
   - For more complex deployment scenarios

### Artifacts

In ADK, **Artifacts** represent a mechanism for managing named, versioned binary data associated with a specific user interaction session or persistently with a user across multiple sessions.

- An Artifact is essentially binary data (like the content of a file) identified by a unique filename
- Artifacts are represented using the `google.genai.types.Part` object
- Storage and retrieval are managed by a dedicated Artifact Service

## Building an AI Agent Team with ADK

Based on the documentation, here's how each role in the requested AI agent team would leverage ADK components:

### 1. Agents Engineer
- Specialized in designing and implementing different types of agents:
  - LLM Agents (using the `LlmAgent` or `Agent` classes)
  - Workflow Agents (using `SequentialAgent`, `ParallelAgent`, `LoopAgent`)
  - Custom Agents (extending `BaseAgent`)
  - Multi-Agent Systems (orchestrating different agent types)
  - Models (selecting and configuring appropriate models)

### 2. Tools Engineer
- Focused on extending agent capabilities through various tools:
  - Function Tools (creating custom Python functions)
  - Built-in Tools (leveraging pre-built ADK tools)
  - Third Party Tools (integrating external libraries)
  - Google Cloud Tools (connecting to Google Cloud services)
  - MCP Tools (implementing Model, Context, Policy tools)
  - OpenAPI Tools (generating tools from API specifications)
  - Authentication (managing secure access to tools)

### 3. Deployment Engineer
- Responsible for packaging and deploying agents:
  - Running Agents (configuring runtime environments)
  - Callbacks (implementing event handlers and monitoring)
  - Deploy (managing deployment to Vertex AI, Cloud Run, or GKE)

### 4. Memory Engineer
- Specialized in managing conversational context:
  - Session (designing session structures)
  - State (implementing state management)
  - Memory (creating cross-session information stores)

### 5. Artifacts Engineer
- Focused on handling binary data:
  - Managing the storage and retrieval of files, images, etc.
  - Implementing appropriate artifact services
  - Defining artifact lifecycles and versioning

## References

- Official ADK Documentation: https://google.github.io/adk-docs/
- GitHub Repository: https://github.com/google/adk-python