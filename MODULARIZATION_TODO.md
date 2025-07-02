# TboardFS Modularization TODO

Based on comprehensive codebase analysis for preparing the project to become a virtual filesystem for TensorBoard data.

## High Priority (Immediate Impact)

### 1. Extract Virtual Filesystem Generator

- **Location:** `efficient_parser.py` lines 1142-1362 (~220 lines)
- **Target:** Create `tboardfs/core/virtual_filesystem.py`
- **Benefit:** Reduces main parser size, better separation of parsing vs. filesystem concerns
- **Details:**
  - Extract `get_virtual_paths()` method and all related helpers
  - Create `VirtualFilesystemGenerator` class
  - Move `_determine_processing_mode()`, `_get_base_directories()` methods
  - Keep clean interface between parser and filesystem generation

### 2. Extract Image Processing Logic

- **Location:** `efficient_parser.py` lines 450-498
- **Target:** Create `tboardfs/core/image_processor.py`
- **Benefit:** Reusable image processing utilities, cleaner main parser
- **Details:**
  - Extract `_decode_image_from_tensor()` method
  - Extract `get_image_extension()` method
  - Create `ImageProcessor` class with static methods
  - Handle tensor-to-image conversion logic

### 3. Create Data Iterator Factory

- **Location:** Scattered throughout `efficient_parser.py` (lines 361-798)
- **Target:** Create `tboardfs/core/data_iterators.py`
- **Benefit:** Eliminates repetitive iterator patterns across 10+ data types
- **Details:**
  - Extract common iteration patterns from `iterate_scalar_data()`, `iterate_image_data()`, etc.
  - Create `DataIteratorFactory` class
  - Implement specialized iterators: `ScalarDataIterator`, `ImageDataIterator`, etc.
  - Reduce code duplication by ~400 lines

## Medium Priority (Architectural Improvement)

### 4. Split Configuration Module

- **Location:** `tboardfs/core/configuration.py` (394 lines)
- **Target:** Split into 3 focused modules
- **Benefit:** More focused configuration classes, better maintainability
- **Details:**
  - Create `tboardfs/core/export_config.py` - Export-specific configuration
  - Create `tboardfs/core/format_config.py` - Format validation and settings
  - Create `tboardfs/core/processing_config.py` - Processing pipeline configuration
  - Maintain backward compatibility with main `TBoardFSConfig` class

### 5. Extract CLI Validation Logic

- **Location:** `tboardfs/cli.py` lines 199-202, 300-303
- **Target:** Create `tboardfs/cli/validators.py`
- **Benefit:** Cleaner CLI command structure, reduced method complexity
- **Details:**
  - Extract repeated validation patterns for image, audio, PLY formats
  - Create `CLIValidator` class with centralized validation methods
  - Reduce CLI method complexity by consolidating validation logic

## Low Priority (Future Virtual Filesystem Features)

### 6. Create Virtual Filesystem Interface Layer

- **Location:** New package structure
- **Target:** Create `tboardfs/filesystem/` package
- **Benefit:** Prepare architecture for FUSE/virtual filesystem implementation
- **Details:**
  - Create `filesystem/interfaces.py` - Virtual filesystem interfaces
  - Create `filesystem/fuse_adapter.py` - Future FUSE filesystem adapter
  - Create `filesystem/cache_manager.py` - Filesystem caching layer
  - Create `filesystem/permissions.py` - Future access control system

### 7. Add Filesystem Metadata System

- **Target:** Extend virtual filesystem with metadata support
- **Benefit:** Support for file attributes, timestamps, permissions in virtual filesystem
- **Details:**
  - Add metadata extraction from TensorBoard events
  - Implement virtual file attributes (size, modification time, etc.)
  - Support for filesystem-like operations (ls, stat, etc.)

## Code Quality Improvements

### 8. Method Decomposition in Parser

- **Location:** `efficient_parser.py` - remaining large methods
- **Target:** Break down methods > 50 lines
- **Details:**
  - Further decompose complex data processing methods
  - Extract helper methods from image/audio/histogram processing
  - Maintain single responsibility principle

### 9. Add Comprehensive Integration Tests

- **Target:** Test module interactions after modularization
- **Details:**
  - Test virtual filesystem generation with various data types
  - Test export pipeline with new modular components
  - Ensure backward compatibility after refactoring

## Architecture Quality Goals

### Current Strengths (Maintain)

- ✅ Excellent use of dataclasses and type hints
- ✅ Good separation of parsing and export concerns
- ✅ Clean abstraction layers (DataSource, BaseExporter)
- ✅ Comprehensive error handling system
- ✅ Well-structured command pattern for CLI
- ✅ Future-ready design patterns

### Technical Debt to Address

- ⚠️ Main parser class is still large (1,362 lines) - **Priority 1-3 address this**
- ⚠️ Configuration module handles too many concerns - **Priority 4 addresses this**
- ⚠️ Missing abstractions for virtual filesystem features - **Priority 6-7 address this**

## Implementation Order

1. **Phase 1:** Items 1-3 (High Priority) - Core parser modularization
1. **Phase 2:** Items 4-5 (Medium Priority) - Configuration and CLI cleanup
1. **Phase 3:** Items 6-7 (Low Priority) - Virtual filesystem preparation
1. **Phase 4:** Items 8-9 - Quality improvements and testing

## Success Metrics

- Main parser reduced to < 800 lines
- No code duplication in data iteration
- Clean separation between parsing, export, and filesystem concerns
- Maintainable configuration system
- Architecture ready for virtual filesystem implementation
- All existing functionality preserved
- Performance maintained or improved

______________________________________________________________________

**Note:** This modularization plan maintains the excellent architectural discipline already demonstrated in the codebase while preparing for future virtual filesystem features. The focus is on extracting the remaining large components to achieve better separation of concerns.
