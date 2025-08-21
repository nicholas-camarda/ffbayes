# Product Documentation

This directory contains product specifications, research, analysis, and strategic planning documents for the fantasy football analytics pipeline.

## Document Organization

### Core Documentation
- **`tech-stack.md`** - **COMPREHENSIVE TECHNICAL DOCUMENTATION** - Complete technical stack, architecture, implementation details, enhancement plans, and next steps
- **`roadmap.md`** - Product roadmap and development timeline
- **`decisions.md`** - Key product decisions and rationale
- **`testing-protocol.md`** - **🚨 MANDATORY TESTING PROTOCOL** - Critical testing mode enforcement and workflow requirements
- **`ORGANIZATION_STRUCTURE.md`** - **📁 PROJECT ORGANIZATION** - Complete folder structure for plots, results, and organized subfolders

## Purpose

These documents serve as:
1. **Technical Reference**: Complete technical implementation and architecture details
2. **Product Planning**: Roadmap and strategic direction
3. **Decision Tracking**: Key decisions and their rationale
4. **Development Guide**: Clear path forward for implementation
5. **Testing Enforcement**: **MANDATORY testing protocol to prevent production mode during testing**

## Usage

- **For Technical Understanding**: Start with `tech-stack.md` for complete technical overview
- **For Product Planning**: Reference `roadmap.md` for current priorities and timeline
- **For Decision Context**: See `decisions.md` for key decisions and rationale
- **For Testing Requirements**: **READ `testing-protocol.md` FIRST** - mandatory testing mode enforcement
- **For Project Organization**: See `ORGANIZATION_STRUCTURE.md` for complete folder structure and file organization

## Maintenance

These documents should be updated as:
- Technical implementations change
- Product specifications evolve
- Strategic priorities shift
- Architecture decisions are made
- Testing protocols are updated

## Quick Reference

- **Complete Technical Overview**: See `tech-stack.md` (contains all implementation details, enhancement plans, and next steps)
- **Current Roadmap**: See `roadmap.md`
- **Key Decisions**: See `decisions.md`
- **🚨 MANDATORY TESTING**: See `testing-protocol.md` - **ALWAYS USE TEST MODE DURING TESTING**
- **📁 Project Organization**: See `ORGANIZATION_STRUCTURE.md` - complete folder structure and file organization

## Critical Testing Requirement

**🚨 BEFORE ANY TESTING OR DEVELOPMENT WORK:**
1. **READ** `testing-protocol.md` completely
2. **SET** `export QUICK_TEST=true`
3. **VERIFY** test mode is active
4. **NEVER** use production mode for testing

**This requirement is non-negotiable and prevents resource waste and unreliable results.**
