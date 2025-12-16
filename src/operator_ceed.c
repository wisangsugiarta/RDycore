#include <ceed/ceed.h>
#include <petscdmceed.h>
#include <private/rdycoreimpl.h>
#include <private/rdyoperatorimpl.h>
#include <private/rdysedimentimpl.h>
#include <private/rdysweimpl.h>

#include "sediment/sediment_ceed_impl.h"
#include "swe/swe_ceed_impl.h"

//-----------------------------
// CEED sub-operators overview
//-----------------------------
//
// The CEED implementation of the shallow water equations consists of two
// composite operators: one for computing fluxes and the other for sources.
// Each of these operators comprises one or more sub-operators, and accepts as
// input a local solution vector, producing the solution's time derivative
// within a global "right hand side" vector.
//
// The flux operator consists of the following sub-operators:
//
// * for the entire domain: an interior flux sub-operator that accepts the local
//   solution vector as input and computes the fluxes on pairs of cells on the
//   interior of the computational domain. This sub-operator is created by the
//   function CreateSWECeedInteriorFluxOperator.
// * for each domain boundary: a boundary flux sub-operator that accepts the
//   local solution vector as input and computes the fluxes into/out of cells
//   adjacent to boundary edges. Each sub-operator is created by the function
//   CreateSWECeedBoundaryFluxOperator.
//
// The source operator consists of the following sub-operators:
//
// * for the entire domain: a source sub-operator that accepts input from the
//   interior and flux sub-operators and adds source terms to produce the
//   time rate of change of the solution vector. This sub-operator is created
//   by the function CreateSWECeedSourceOperator.
//
// The relevant function documentation includes a comprehensive list of input
// and output fields for each sub-operator.

// CEED uses C99 VLA features for shaping multidimensional
// arrays, which don't have the same drawbacks as VLA allocations.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wvla"

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wvla"

static PetscErrorCode CreateInteriorFluxQFunction(Ceed ceed, const RDyConfig config, CeedQFunction *qf) {
  PetscFunctionBeginUser;

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;

  CeedQFunctionContext qf_context;
  if (num_sediment_comp == 0) {  // flow only, and SWE is it!
    PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEFlux_Roe, SWEFlux_Roe_loc, qf));
    PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
  } else {
    PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentFlux_Roe, SedimentFlux_Roe_loc, qf));
    PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
  }

  // add the context to the Q function
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(*qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CEED operator for solving governing equations by computing
/// fluxes on interior edges
/// Creates a CeedOperator that computes fluxes between pairs of cells on the
/// domain's interior.
///
/// Active input fields:
///    * `q_left[num_interior_edges][3]` - an array associating a 3-DOF left cell
///      input state with each edge separating two interior cells
///    * `q_right[num_interior_edges][3]` - an array associating a 3-DOF right cell
///      input state with each edge separating two interior cells
///
/// Passive input fields:
///    * `geom[num_interior_edges][4]` - an array associating 4 geometric factors
///      with each edge separating two interior cells:
///        1. sin(theta), where theta is the angle between the edge and the y axis
///        2. cos(theta), where theta is the angle between the edge and the y axis
///        3. -L / A_l, where L is the edge's length and A_l is the area of the "left" cell
///        4. L / A_r, where L is the edge's length and A_r is the area of the "right" cell
///
/// Active output fields:
///    * `cell_left[num_interior_edges][3]` - an array associating a 3-DOF left cell
///      output state with each edge separating two interior cells
///    * `cell_right[num_interior_edges][3]` - an array associating a 3-DOF right cell
///      output state with each edge separating two interior cells
///
/// Passive output fields:
///    * `flux[num_owned_cells][3]` - an array associating riemann fluxes
///      with each owned cell
///    * `courant_number[num_interior_edges][1]` - an array associating the
///      Courant number (max wave speed) with each edge separating two interior
///      cells
///
/// Q-function context field labels:
///    * `time step` - the time step used by the operator
///    * `small h value` - the water height below which dry conditions are assumed
///    * `gravity` - the acceleration due to gravity [m/s/s]
///
/// @param [in]  config  RDycore's configuration
/// @param [in]  mesh    mesh defining the computational domain of the operator
/// @param [out] ceed_op a CeedOperator that is created and returned
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateCeedInteriorFluxOperator(const RDyConfig config, RDyMesh *mesh, CeedOperator *ceed_op) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;

  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  CeedQFunction qf;
  PetscCall(CreateInteriorFluxQFunction(ceed, config, &qf));

  // add inputs and outputs
  // NOTE: the order in which these inputs and outputs are specified determines
  // NOTE: their indexing within the Q-function's implementation (swe_ceed_impl.h)
  CeedInt num_comp_geom = 4, num_comp_cnum = 2;
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_right", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_right", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "courant_number", num_comp_cnum, CEED_EVAL_NONE));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction q_restrict_l, q_restrict_r, c_restrict_l, c_restrict_r, restrict_geom, restrict_flux, restrict_cnum;
  CeedVector          geom, flux, cnum;
  {
    CeedInt num_edges = mesh->num_owned_internal_edges;

    // create a vector of geometric factors that transform fluxes to cell states
    CeedInt g_strides[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_geom, num_edges * num_comp_geom, g_strides, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));
    CeedScalar(*g)[4];
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l        = edges->cell_ids[2 * iedge];
      CeedInt r        = edges->cell_ids[2 * iedge + 1];
      g[owned_edge][0] = edges->sn[iedge];
      g[owned_edge][1] = edges->cn[iedge];
      g[owned_edge][2] = -edges->lengths[iedge] / cells->areas[l];
      g[owned_edge][3] = edges->lengths[iedge] / cells->areas[r];
      owned_edge++;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

    // create a vector to store inter-cell fluxes
    CeedInt f_strides[] = {num_comp, 1, num_comp};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp, num_edges * num_comp, f_strides, &restrict_flux));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_flux, &flux, NULL));
    PetscCallCEED(CeedVectorSetValue(flux, 0.0));

    // create a vector to store the courant number for each edge
    CeedInt cnum_strides[] = {num_comp_cnum, 1, num_comp_cnum};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_cnum, num_edges * num_comp_cnum, cnum_strides, &restrict_cnum));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_cnum, &cnum, NULL));
    PetscCallCEED(CeedVectorSetValue(cnum, 0.0));

    // create element restrictions for (active) left and right input/output states
    CeedInt *q_offset_l, *q_offset_r, *c_offset_l, *c_offset_r;
    PetscCall(PetscMalloc2(num_edges, &q_offset_l, num_edges, &q_offset_r));
    PetscCall(PetscMalloc2(num_edges, &c_offset_l, num_edges, &c_offset_r));
    for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l              = edges->cell_ids[2 * iedge];
      CeedInt r              = edges->cell_ids[2 * iedge + 1];
      q_offset_l[owned_edge] = l * num_comp;
      q_offset_r[owned_edge] = r * num_comp;
      c_offset_l[owned_edge] = cells->local_to_owned[l] * num_comp;
      c_offset_r[owned_edge] = cells->local_to_owned[r] * num_comp;
      owned_edge++;
    }
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, q_offset_l,
                                            &q_restrict_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, q_offset_r,
                                            &q_restrict_r));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, c_offset_l,
                                            &c_restrict_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES, c_offset_r,
                                            &c_restrict_r));
    PetscCall(PetscFree2(q_offset_l, q_offset_r));
    PetscCall(PetscFree2(c_offset_l, c_offset_r));
    if (0) {
      PetscCallCEED(CeedElemRestrictionView(q_restrict_l, stdout));
      PetscCallCEED(CeedElemRestrictionView(q_restrict_r, stdout));
      PetscCallCEED(CeedElemRestrictionView(c_restrict_l, stdout));
      PetscCallCEED(CeedElemRestrictionView(c_restrict_r, stdout));
    }
  }

  // create the operator itself and assign its active/passive inputs/outputs
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, ceed_op));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "geom", restrict_geom, CEED_BASIS_NONE, geom));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q_left", q_restrict_l, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q_right", q_restrict_r, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell_left", c_restrict_l, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell_right", c_restrict_r, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "flux", restrict_flux, CEED_BASIS_NONE, flux));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "courant_number", restrict_cnum, CEED_BASIS_NONE, cnum));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_flux));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_cnum));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_r));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_r));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&flux));
  PetscCallCEED(CeedVectorDestroy(&cnum));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

static PetscErrorCode BuildCellNeighborConnectivity(RDyMesh *mesh) {
  PetscFunctionBeginUser;

  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  // First, reset neighbor counts to 0
  for (CeedInt c = 0; c < mesh->num_cells; c++) {
    cells->num_neighbors[c] = 0;
  }

  // count neighbors by examining edges
  for (CeedInt e = 0; e < mesh->num_edges; e++) {
    CeedInt left = edges->cell_ids[2 * e];
    CeedInt right = edges->cell_ids[2 * e + 1];

    // Valid cells - no boundary)
    if (left >= 0 && left < mesh->num_cells && right >= 0 && right < mesh->num_cells) {
      cells->num_neighbors[left]++;
      cells->num_neighbors[right]++;
    }
  }

  // Compute offsets
  cells->neighbor_offsets[0] = 0;
  for (CeedInt c = 1; c < mesh->num_cells; c++) {
    cells->neighbor_offsets[c] = cells->neighbor_offsets[c-1] + cells->num_neighbors[c-1];
  }

  // Reset counts
  CeedInt *neighbor_counters;
  PetscCall(PetscCalloc1(mesh->num_cells, &neighbor_counters));

  for (CeedInt c = 0; c < mesh->num_cells; c++) {
    neighbor_counters[c] = 0;
  }

  // populate the actual neighbor IDs
  for (CeedInt e = 0; e < mesh->num_edges; e++) {
    CeedInt left = edges->cell_ids[2 * e];
    CeedInt right = edges->cell_ids[2 * e + 1];

    if (left >= 0 && left < mesh->num_cells && right >= 0 && right < mesh->num_cells) {
      // Add right as neighbor of left
      CeedInt left_idx = cells->neighbor_offsets[left] + neighbor_counters[left];
      cells->neighbor_ids[left_idx] = right;
      neighbor_counters[left]++;

      // Add left as neighbor of right
      CeedInt right_idx = cells->neighbor_offsets[right] + neighbor_counters[right];
      cells->neighbor_ids[right_idx] = left;
      neighbor_counters[right]++;
    }
  }

  PetscCall(PetscFree(neighbor_counters));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode CreateCeedInteriorFluxOperatorReconstruction(const RDyConfig config, RDyMesh *mesh, CeedOperator *ceed_op) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  // Must match the actual solution vector layout!
  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;


  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  // Build cell neighbor connectivity
  PetscCall(BuildCellNeighborConnectivity(mesh));

  // Create Q-function
  CeedQFunction qf;
  PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEFluxReconstruction_Roe, SWEFluxReconstruction_Roe_loc, &qf));

  CeedQFunctionContext qf_context;
  PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
  PetscCallCEED(CeedQFunctionSetContext(qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  CeedInt num_edges = mesh->num_owned_internal_edges;

  // Component sizes
  CeedInt num_comp_edge_data = 5;
  CeedInt num_comp_cell_data = 6;
  CeedInt num_comp_neighbor_coords = 16;
  CeedInt num_comp_neighbor_values = 24;
  CeedInt num_comp_cnum = 2;

  // Q-function inputs
  PetscCallCEED(CeedQFunctionAddInput(qf, "edge_data", num_comp_edge_data, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "cell_data", num_comp_cell_data, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "neighbor_coords", num_comp_neighbor_coords, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "neighbor_values", num_comp_neighbor_values, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_left", num_flow_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_right", num_flow_comp, CEED_EVAL_NONE));

  // Q-function outputs
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_left", num_flow_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_right", num_flow_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "flux", num_flow_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "courant_number", num_comp_cnum, CEED_EVAL_NONE));

  // Create vectors and restrictions
  CeedElemRestriction restrict_edge_data, restrict_cell_data, restrict_neighbor_coords, restrict_neighbor_values;
  CeedElemRestriction restrict_q_l, restrict_q_r, restrict_cell_l, restrict_cell_r;
  CeedElemRestriction restrict_flux, restrict_cnum;
  CeedVector edge_data, cell_data, neighbor_coords, flux, cnum;

  // 1. Edge data (static)
  {
    CeedInt strides[] = {num_comp_edge_data, 1, num_comp_edge_data};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_edge_data,
                                                   num_edges * num_comp_edge_data, strides, &restrict_edge_data));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_edge_data, &edge_data, NULL));
    PetscCallCEED(CeedVectorSetValue(edge_data, 0.0));

    CeedScalar(*ed)[5];
    PetscCallCEED(CeedVectorGetArray(edge_data, CEED_MEM_HOST, (CeedScalar **)&ed));
    for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;

      ed[owned_edge][0] = edges->sn[iedge];
      ed[owned_edge][1] = edges->cn[iedge];
      ed[owned_edge][2] = edges->lengths[iedge];
      ed[owned_edge][3] = edges->centroids[iedge].X[0];
      ed[owned_edge][4] = edges->centroids[iedge].X[1];

      owned_edge++;
    }
    PetscCallCEED(CeedVectorRestoreArray(edge_data, (CeedScalar **)&ed));
  }

  // 2. Cell data per edge (static)
  {
    CeedInt strides[] = {num_comp_cell_data, 1, num_comp_cell_data};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_cell_data,
                                                   num_edges * num_comp_cell_data, strides, &restrict_cell_data));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_cell_data, &cell_data, NULL));
    PetscCallCEED(CeedVectorSetValue(cell_data, 0.0));

    CeedScalar(*cd)[6];
    PetscCallCEED(CeedVectorGetArray(cell_data, CEED_MEM_HOST, (CeedScalar **)&cd));
    for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;

      CeedInt l = edges->cell_ids[2 * iedge];
      CeedInt r = edges->cell_ids[2 * iedge + 1];

      cd[owned_edge][0] = cells->centroids[l].X[0];
      cd[owned_edge][1] = cells->centroids[l].X[1];
      cd[owned_edge][2] = cells->areas[l];
      cd[owned_edge][3] = cells->centroids[r].X[0];
      cd[owned_edge][4] = cells->centroids[r].X[1];
      cd[owned_edge][5] = cells->areas[r];

      owned_edge++;
    }
    PetscCallCEED(CeedVectorRestoreArray(cell_data, (CeedScalar **)&cd));
  }

  // 3. Neighbor coordinates per edge (static)
  {
    CeedInt strides[] = {num_comp_neighbor_coords, 1, num_comp_neighbor_coords};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_neighbor_coords,
                                                   num_edges * num_comp_neighbor_coords, strides, &restrict_neighbor_coords));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_neighbor_coords, &neighbor_coords, NULL));
    PetscCallCEED(CeedVectorSetValue(neighbor_coords, 0.0));

    CeedScalar *nc_data;
    PetscCallCEED(CeedVectorGetArray(neighbor_coords, CEED_MEM_HOST, &nc_data));

    for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;

      CeedInt l = edges->cell_ids[2 * iedge];
      CeedInt r = edges->cell_ids[2 * iedge + 1];

      CeedInt base_idx = owned_edge * num_comp_neighbor_coords;

      // Pack neighbor coordinates for left cell (first 8 components: 4 neighbors * 2 coords)
      CeedInt valid_left_neighbors = 0;
      for (CeedInt n = 0; n < cells->num_neighbors[l] && valid_left_neighbors < 4; n++) {
        CeedInt neighbor = cells->neighbor_ids[cells->neighbor_offsets[l] + n];

        // Skip boundary neighbors (ID = -1)
        if (neighbor >= 0 && neighbor < mesh->num_cells) {
          nc_data[base_idx + 2*valid_left_neighbors] = cells->centroids[neighbor].X[0];
          nc_data[base_idx + 2*valid_left_neighbors + 1] = cells->centroids[neighbor].X[1];
          valid_left_neighbors++;
        }
      }

      // Fill remaining slots with zeros
      for (CeedInt n = valid_left_neighbors; n < 4; n++) {
        nc_data[base_idx + 2*n] = 0.0;
        nc_data[base_idx + 2*n + 1] = 0.0;
      }

      // Pack neighbor coordinates for right cell (last 8 components)
      CeedInt valid_right_neighbors = 0;
      for (CeedInt n = 0; n < cells->num_neighbors[r] && valid_right_neighbors < 4; n++) {
        CeedInt neighbor = cells->neighbor_ids[cells->neighbor_offsets[r] + n];

        if (neighbor >= 0 && neighbor < mesh->num_cells) {
          nc_data[base_idx + 8 + 2*valid_right_neighbors] = cells->centroids[neighbor].X[0];
          nc_data[base_idx + 8 + 2*valid_right_neighbors + 1] = cells->centroids[neighbor].X[1];
          valid_right_neighbors++;
        }
      }

      // Fill remaining slots with zeros
      for (CeedInt n = valid_right_neighbors; n < 4; n++) {
        nc_data[base_idx + 8 + 2*n] = 0.0;
        nc_data[base_idx + 8 + 2*n + 1] = 0.0;
      }

      owned_edge++;
    }

    PetscCallCEED(CeedVectorRestoreArray(neighbor_coords, &nc_data));
  }

  // 4. Create neighbor values restriction (active) - layout for [comp][edge] access
  {
    CeedInt *neighbor_offsets;
    PetscCall(PetscMalloc1(num_edges * num_comp_neighbor_values, &neighbor_offsets));

    CeedInt max_valid_offset = mesh->num_cells * num_comp;
    CeedInt out_of_bounds_count = 0;

    // Pack data in order for [component][edge] access pattern
    for (CeedInt comp_idx = 0; comp_idx < num_comp_neighbor_values; comp_idx++) {
      for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
        CeedInt iedge = edges->internal_edge_ids[e];
        if (!edges->is_owned[iedge]) continue;

        CeedInt l = edges->cell_ids[2 * iedge];
        CeedInt r = edges->cell_ids[2 * iedge + 1];

        // Validate left and right cell IDs
        if (l < 0 || l >= mesh->num_cells || r < 0 || r >= mesh->num_cells) {
          PetscCall(PetscFree(neighbor_offsets));
          PetscFunctionReturn(PETSC_ERR_ARG_OUTOFRANGE);
        }

        CeedInt cell_id, flow_comp;

        // Determine which cell and component this comp_idx represents
        if (comp_idx < 12) {
          // Left cell neighbors (components 0-11)
          CeedInt neighbor_idx = comp_idx / 3;  // which neighbor (0-3)
          flow_comp = comp_idx % 3;              // which flow component (0-2)

          if (neighbor_idx < cells->num_neighbors[l]) {
            cell_id = cells->neighbor_ids[cells->neighbor_offsets[l] + neighbor_idx];
            if (cell_id < 0 || cell_id >= mesh->num_cells) {
              cell_id = l;  // fallback to left cell's own data
            }
          } else {
            cell_id = l;  // padding: use left cell's own data
          }
        } else {
          // Right cell neighbors (components 12-23)
          CeedInt neighbor_idx = (comp_idx - 12) / 3;  // which neighbor (0-3)
          flow_comp = (comp_idx - 12) % 3;              // which flow component (0-2)

          if (neighbor_idx < cells->num_neighbors[r]) {
            cell_id = cells->neighbor_ids[cells->neighbor_offsets[r] + neighbor_idx];
            if (cell_id < 0 || cell_id >= mesh->num_cells) {
              cell_id = r;  // fallback to right cell's own data
            }
          } else {
            cell_id = r;  // padding: use right cell's own data
          }
        }

        CeedInt offset = cell_id * num_comp + flow_comp;
        if (offset >= max_valid_offset) {
          out_of_bounds_count++;
        }

        neighbor_offsets[comp_idx * num_edges + owned_edge] = offset;

        owned_edge++;
      }
    }

    if (out_of_bounds_count > 0) {
      PetscCall(PetscFree(neighbor_offsets));
      PetscFunctionReturn(PETSC_ERR_ARG_OUTOFRANGE);
    }

    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_comp_neighbor_values, 1,
                                            mesh->num_cells * num_comp, CEED_MEM_HOST,
                                            CEED_COPY_VALUES, neighbor_offsets, &restrict_neighbor_values));
    PetscCall(PetscFree(neighbor_offsets));
  }

  // 5. Create input restrictions for left/right solutions
  {
    CeedInt *q_offset_l, *q_offset_r;
    PetscCall(PetscMalloc2(num_edges, &q_offset_l, num_edges, &q_offset_r));

    for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l = edges->cell_ids[2 * iedge];
      CeedInt r = edges->cell_ids[2 * iedge + 1];
      q_offset_l[owned_edge] = l * num_comp;
      q_offset_r[owned_edge] = r * num_comp;
      owned_edge++;
    }

    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_flow_comp, 1,
                                            mesh->num_cells * num_comp, CEED_MEM_HOST,
                                            CEED_COPY_VALUES, q_offset_l, &restrict_q_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_flow_comp, 1,
                                            mesh->num_cells * num_comp, CEED_MEM_HOST,
                                            CEED_COPY_VALUES, q_offset_r, &restrict_q_r));
    PetscCall(PetscFree2(q_offset_l, q_offset_r));
  }

  // 6. Output restrictions
  {
    CeedInt *c_offset_l, *c_offset_r;
    PetscCall(PetscMalloc2(num_edges, &c_offset_l, num_edges, &c_offset_r));

    for (CeedInt e = 0, owned_edge = 0; e < mesh->num_internal_edges; e++) {
      CeedInt iedge = edges->internal_edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l = edges->cell_ids[2 * iedge];
      CeedInt r = edges->cell_ids[2 * iedge + 1];
      c_offset_l[owned_edge] = cells->local_to_owned[l] * num_comp;
      c_offset_r[owned_edge] = cells->local_to_owned[r] * num_comp;
      owned_edge++;
    }

    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_flow_comp, 1,
                                            mesh->num_cells * num_comp, CEED_MEM_HOST,
                                            CEED_COPY_VALUES, c_offset_l, &restrict_cell_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_edges, 1, num_flow_comp, 1,
                                            mesh->num_cells * num_comp, CEED_MEM_HOST,
                                            CEED_COPY_VALUES, c_offset_r, &restrict_cell_r));
    PetscCall(PetscFree2(c_offset_l, c_offset_r));
  }

  // 7. Flux and courant output vectors
  {
    CeedInt f_strides[] = {num_flow_comp, 1, num_flow_comp};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_flow_comp,
                                                   num_edges * num_flow_comp, f_strides, &restrict_flux));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_flux, &flux, NULL));
    PetscCallCEED(CeedVectorSetValue(flux, 0.0));

    CeedInt c_strides[] = {num_comp_cnum, 1, num_comp_cnum};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_edges, 1, num_comp_cnum,
                                                   num_edges * num_comp_cnum, c_strides, &restrict_cnum));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_cnum, &cnum, NULL));
    PetscCallCEED(CeedVectorSetValue(cnum, 0.0));
  }

  // 8. Create operator and set fields
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, ceed_op));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "edge_data", restrict_edge_data, CEED_BASIS_NONE, edge_data));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell_data", restrict_cell_data, CEED_BASIS_NONE, cell_data));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "neighbor_coords", restrict_neighbor_coords, CEED_BASIS_NONE, neighbor_coords));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "neighbor_values", restrict_neighbor_values, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q_left", restrict_q_l, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q_right", restrict_q_r, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell_left", restrict_cell_l, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell_right", restrict_cell_r, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "flux", restrict_flux, CEED_BASIS_NONE, flux));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "courant_number", restrict_cnum, CEED_BASIS_NONE, cnum));

  // 9. Cleanup
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_edge_data));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_cell_data));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_neighbor_coords));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_neighbor_values));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_q_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_q_r));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_cell_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_cell_r));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_flux));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_cnum));
  PetscCallCEED(CeedVectorDestroy(&edge_data));
  PetscCallCEED(CeedVectorDestroy(&cell_data));
  PetscCallCEED(CeedVectorDestroy(&neighbor_coords));
  PetscCallCEED(CeedVectorDestroy(&flux));
  PetscCallCEED(CeedVectorDestroy(&cnum));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode CreateBoundaryFluxQFunction(Ceed ceed, const RDyConfig config, RDyBoundary boundary, RDyCondition boundary_condition,
                                                  CeedQFunction *qf) {
  PetscFunctionBeginUser;

  int num_sediment_comp = config.physics.sediment.num_classes;

  CeedQFunctionContext qf_context;
  switch (boundary_condition.flow->type) {
    case CONDITION_DIRICHLET:
      if (num_sediment_comp == 0) {  // flow only
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEBoundaryFlux_Dirichlet_Roe, SWEBoundaryFlux_Dirichlet_Roe_loc, qf));
        PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
      } else {  // sediment dynamics
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentBoundaryFlux_Dirichlet_Roe, SedimentBoundaryFlux_Dirichlet_Roe_loc, qf));
        PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
      }
      break;
    case CONDITION_REFLECTING:
      if (num_sediment_comp == 0) {  // flow only
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEBoundaryFlux_Reflecting_Roe, SWEBoundaryFlux_Reflecting_Roe_loc, qf));
        PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
      } else {  // sediment dynamics
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentBoundaryFlux_Reflecting_Roe, SedimentBoundaryFlux_Reflecting_Roe_loc, qf));
        PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
      }
      break;
    case CONDITION_CRITICAL_OUTFLOW:
      if (num_sediment_comp == 0) {  // flow only
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWEBoundaryFlux_Outflow_Roe, SWEBoundaryFlux_Outflow_Roe_loc, qf));
        PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
      } else {  // sediment dynamics
        PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "CONDITION_CRITICAL_OUTFLOW not implemented");
      }
      break;
    default:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Invalid boundary condition encountered for boundary %" PetscInt_FMT "\n",
                 boundary.id);
  }

  // add the context to the Q function
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(*qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CEED operator that computes fluxes through edges on the boundary of a domain.
/// Creates a CeedOperator that computes fluxes through edges on the boundary
/// of a domain.
///
/// Active input fields:
///    * `q_left[num_boundary_edges][3]` - an array associating a 3-DOF left
///      (interior) cell input state with each boundary edge
///
/// Passive input fields:
///    * `geom[num_boundary_edges][3]` - an array associating 3 geometric factors
///      with each boundary edge:
///        1. sin(theta), where theta is the angle between the edge and the y axis
///        2. cos(theta), where theta is the angle between the edge and the y axis
///        3. -L / A_l, where L is the edge's length and A_l is the area of the
///           "left" (interior) cell
///    * `q_dirichlet[num_boundary_edges][3]` - an array associating 3 boundary
///      values with each boundary edge (**iff the boundary associated with the
///      sub-operator is assigned a dirichlet boundary condition**)
///
/// Active output fields:
///    * `cell_left[num_boundary_edges][3]` - an array associating a 3-DOF left
///      (interior) cell output state with each boundary edge
///
/// Passive output fields:
///    * `flux[num_boundary_edges][3]` - an array associating riemann fluxes
///      with each boundary edge (to be summed to compute their divergence)
///    * `courant_number[num_interior_edges][1]` - an array associating the
///      Courant number (max wave speed) with each boundary edge
///
/// Q-function context field labels:
///    * `time step` - the time step used by the operator
///    * `small h value` - the water height below which dry conditions are assumed
///    * `gravity` - the acceleration due to gravity [m/s/s]
///
/// @param [in]  config             RDycore's configuration
/// @param [in]  mesh               mesh defining the computational domain of the operator
/// @param [in]  boundary           a RDyBoundary struct describing the boundary on which the boundary condition is applied
/// @param [in]  boundary_condition a RDyCondition describing the type of boundary condition
/// @param [out] ceed_op            a CeedOperator that is created and returned
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedBoundaryFluxOperator(const RDyConfig config, RDyMesh *mesh, RDyBoundary boundary, RDyCondition boundary_condition,
                                              CeedOperator *ceed_op) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;

  RDyCells *cells = &mesh->cells;
  RDyEdges *edges = &mesh->edges;

  CeedQFunction qf;
  PetscCall(CreateBoundaryFluxQFunction(ceed, config, boundary, boundary_condition, &qf));

  // add inputs/outputs
  // NOTE: the order in which these inputs and outputs are specified determines
  // NOTE: their indexing within the Q-function's implementation
  CeedInt num_comp_geom = 3, num_comp_cnum = 1;
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q_left", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell_left", num_comp, CEED_EVAL_NONE));
  if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
    PetscCallCEED(CeedQFunctionAddInput(qf, "q_dirichlet", num_comp, CEED_EVAL_NONE));
  }
  PetscCallCEED(CeedQFunctionAddOutput(qf, "flux", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "flux_accumulated", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "courant_number", num_comp_cnum, CEED_EVAL_NONE));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction q_restrict_l, c_restrict_l, restrict_dirichlet, restrict_geom, restrict_flux, restrict_cnum;
  CeedVector          geom, flux, flux_accumulated, dirichlet, cnum;
  {
    CeedInt num_edges = boundary.num_edges;

    // create element restrictions for left and right input/output states
    CeedInt *q_offset_l, *c_offset_l, *offset_dirichlet = NULL;
    PetscCall(PetscMalloc1(num_edges, &q_offset_l));
    PetscCall(PetscMalloc1(num_edges, &c_offset_l));
    if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
      PetscCall(PetscMalloc1(num_edges, &offset_dirichlet));
    }

    // create a vector of geometric factors that transform fluxes to cell states
    CeedInt num_owned_edges = 0;
    for (CeedInt e = 0; e < boundary.num_edges; e++) {
      CeedInt iedge = boundary.edge_ids[e];
      if (edges->is_owned[iedge]) num_owned_edges++;
    }
    CeedInt g_strides[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_edges, 1, num_comp_geom, num_edges * num_comp_geom, g_strides, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));
    CeedScalar(*g)[3];
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt e = 0, owned_edge = 0; e < num_edges; e++) {
      CeedInt iedge = boundary.edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l        = edges->cell_ids[2 * iedge];
      g[owned_edge][0] = edges->sn[iedge];
      g[owned_edge][1] = edges->cn[iedge];
      g[owned_edge][2] = -edges->lengths[iedge] / cells->areas[l];
      owned_edge++;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

    // create a vector to store accumulated fluxes (flux divergences)
    CeedInt f_strides[] = {num_comp, 1, num_comp};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_edges, 1, num_comp, num_edges * num_comp, f_strides, &restrict_flux));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_flux, &flux, NULL));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_flux, &flux_accumulated, NULL));
    PetscCallCEED(CeedVectorSetValue(flux, 0.0));
    PetscCallCEED(CeedVectorSetValue(flux_accumulated, 0.0));

    // create a vector to store the courant number for each edge
    CeedInt cnum_strides[] = {num_comp_cnum, 1, num_comp_cnum};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_edges, 1, num_comp_cnum, num_edges * num_comp_cnum, cnum_strides, &restrict_cnum));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_cnum, &cnum, NULL));
    PetscCallCEED(CeedVectorSetValue(cnum, 0.0));

    // create an element restriction for the (active) "left" (interior) input/output states
    for (CeedInt e = 0, owned_edge = 0; e < num_edges; e++) {
      CeedInt iedge = boundary.edge_ids[e];
      if (!edges->is_owned[iedge]) continue;
      CeedInt l              = edges->cell_ids[2 * iedge];
      q_offset_l[owned_edge] = l * num_comp;
      c_offset_l[owned_edge] = cells->local_to_owned[l] * num_comp;
      if (offset_dirichlet) {  // Dirichlet boundary values
        offset_dirichlet[owned_edge] = e * num_comp;
      }
      owned_edge++;
    }
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            q_offset_l, &q_restrict_l));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, mesh->num_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            c_offset_l, &c_restrict_l));
    PetscCall(PetscFree(q_offset_l));
    PetscCall(PetscFree(c_offset_l));
    if (0) {
      PetscCallCEED(CeedElemRestrictionView(q_restrict_l, stdout));
      PetscCallCEED(CeedElemRestrictionView(c_restrict_l, stdout));
    }

    // create a vector to store (Dirichlet) boundary values if needed
    if (offset_dirichlet) {
      PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_edges, 1, num_comp, 1, num_edges * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                              offset_dirichlet, &restrict_dirichlet));
      PetscCall(PetscFree(offset_dirichlet));
      if (0) PetscCallCEED(CeedElemRestrictionView(restrict_dirichlet, stdout));
      PetscCallCEED(CeedElemRestrictionCreateVector(restrict_dirichlet, &dirichlet, NULL));
      PetscCallCEED(CeedVectorSetValue(dirichlet, 0.0));
    }
  }

  // create the operator itself and assign its active/passive inputs/outputs
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, ceed_op));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "geom", restrict_geom, CEED_BASIS_NONE, geom));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q_left", q_restrict_l, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  if (boundary_condition.flow->type == CONDITION_DIRICHLET) {
    PetscCallCEED(CeedOperatorSetField(*ceed_op, "q_dirichlet", restrict_dirichlet, CEED_BASIS_NONE, dirichlet));
  }
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell_left", c_restrict_l, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "flux", restrict_flux, CEED_BASIS_NONE, flux));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "flux_accumulated", restrict_flux, CEED_BASIS_NONE, flux_accumulated));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "courant_number", restrict_cnum, CEED_BASIS_NONE, cnum));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_flux));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_cnum));
  PetscCallCEED(CeedElemRestrictionDestroy(&q_restrict_l));
  PetscCallCEED(CeedElemRestrictionDestroy(&c_restrict_l));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&flux));
  PetscCallCEED(CeedVectorDestroy(&flux_accumulated));
  PetscCallCEED(CeedVectorDestroy(&cnum));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

/// Creates a CEED flux operator appropriate for the given configuration.
/// @param [in]    config              the configuration defining the physics and numerics for the new operator
/// @param [in]    mesh                a mesh containing geometric and topological information for the domain
/// @param [in]    num_boundaries      the number of distinct boundaries bounding the computational domain
/// @param [in]    boundaries          an array of distinct boundaries bounding the computational domain
/// @param [in]    boundary_conditions an array of boundary conditions corresponding to the domain boundaries
/// @param [out]   flux_op             the newly created operator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedFluxOperator(RDyConfig *config, RDyMesh *mesh, PetscInt num_boundaries, RDyBoundary *boundaries,
                                      RDyCondition *boundary_conditions, CeedOperator *flux_op) {
  PetscFunctionBegin;

  Ceed ceed = CeedContext();

  PetscCall(CeedCompositeOperatorCreate(ceed, flux_op));

  if (config->physics.flow.mode != FLOW_SWE) {
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "SWE is the only supported flow model!");
  }

  // flux suboperator 0: fluxes between interior cells

  CeedOperator interior_flux_op;
  PetscCall(CreateCeedInteriorFluxOperator(*config, mesh, &interior_flux_op));
  PetscCall(CeedCompositeOperatorAddSub(*flux_op, interior_flux_op));

  // flux suboperators 1 to num_boundaries: fluxes on boundary edges
  for (CeedInt b = 0; b < num_boundaries; ++b) {
    CeedOperator boundary_flux_op;
    RDyBoundary  boundary  = boundaries[b];
    RDyCondition condition = boundary_conditions[b];
    PetscCall(CreateCeedBoundaryFluxOperator(*config, mesh, boundary, condition, &boundary_flux_op));
    PetscCall(CeedCompositeOperatorAddSub(*flux_op, boundary_flux_op));
  }
  PetscFunctionReturn(PETSC_SUCCESS);
}
PetscErrorCode CreateCeedFluxOperatorReconstructed(RDyConfig *config, RDyMesh *mesh, 
                                                   PetscInt num_boundaries, RDyBoundary *boundaries,
                                                   RDyCondition *boundary_conditions, CeedOperator *flux_op) {
  PetscFunctionBegin;

  Ceed ceed = CeedContext();

  PetscCall(CeedCompositeOperatorCreate(ceed, flux_op));

  if (config->physics.flow.mode != FLOW_SWE) {
    PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "SWE is the only supported flow model!");
  }
  printf("DEBUG: Using slope reconstruction at Ceed Flux Op\n");
  // Create interior flux operator with reconstruction
  CeedOperator interior_flux_op;
  PetscCall(CreateCeedInteriorFluxOperatorReconstruction(*config, mesh, &interior_flux_op));
  PetscCall(CeedCompositeOperatorAddSub(*flux_op, interior_flux_op));

  // Could be enhanced later to use reconstructed values at boundaries too
  for (CeedInt b = 0; b < num_boundaries; ++b) {
    CeedOperator boundary_flux_op;
    RDyBoundary  boundary  = boundaries[b];
    RDyCondition condition = boundary_conditions[b];
    PetscCall(CreateCeedBoundaryFluxOperator(*config, mesh, boundary, condition, &boundary_flux_op));
    PetscCall(CeedCompositeOperatorAddSub(*flux_op, boundary_flux_op));
  }

  PetscFunctionReturn(PETSC_SUCCESS);
}


static PetscErrorCode CreateSourceQFunction(Ceed ceed, const RDyConfig config, CeedQFunction *qf) {
  PetscFunctionBeginUser;
  CeedInt num_sediment_comp = config.physics.sediment.num_classes;

  CeedQFunctionContext qf_context;
  switch (config.physics.flow.source.method) {
    case SOURCE_SEMI_IMPLICIT:
      if (num_sediment_comp == 0) {  // flow only
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWESourceTermSemiImplicit, SWESourceTermSemiImplicit_loc, qf));
        PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
      } else {
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SedimentSourceTermSemiImplicit, SedimentSourceTermSemiImplicit_loc, qf));
        PetscCall(CreateSedimentQFunctionContext(ceed, config, &qf_context));
      }
      break;
    case SOURCE_IMPLICIT_XQ2018:
      if (num_sediment_comp == 0) {  // flow only
        PetscCallCEED(CeedQFunctionCreateInterior(ceed, 1, SWESourceTermImplicitXQ2018, SWESourceTermImplicitXQ2018_loc, qf));
        PetscCall(CreateSWEQFunctionContext(ceed, config, &qf_context));
      } else {
        PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "SOURCE_IMPLICIT_XQ2018 is not supported in sediment CEED version");
      }
      break;
    default:
      PetscCheck(PETSC_FALSE, PETSC_COMM_WORLD, PETSC_ERR_USER, "Only semi_implicit source-term is supported in the CEED version");
      break;
  }

  // add the context to the Q function
  if (0) PetscCallCEED(CeedQFunctionContextView(qf_context, stdout));
  PetscCallCEED(CeedQFunctionSetContext(*qf, qf_context));
  PetscCallCEED(CeedQFunctionContextDestroy(&qf_context));

  PetscFunctionReturn(PETSC_SUCCESS);
}

/// @brief Creates a CEED operator for computing source terms within a domain.
/// Creates a CeedOperator that computes sources for a domain.
///
/// Active input fields:
///    * `q[num_owned_cells][3]` - an array associating a 3-DOF solution input
///      state with each (owned) cell in the domain
///
/// Passive input fields:
///    * `geom[num_owned_cells][2]` - an array associating 2 geometric factors
///      with each (owned) cell in the domain:
///        1. dz/dx, the derivative of the elevation function z(x, y) w.r.t. x,
///           evaluated at the cell center
///        2. dz/dy, the derivative of the elevation function z(x, y) w.r.t. y,
///           evaluated at the cell center
///    * `mat_props[num_owned_cells][N]` - an array associating material
///      properties with each (owned) cell in the domain
///    * `riemannf[num_owned_cells][3]` - an array associating a 3-component
///      flux divergence with each (owned) cell in the domain
///    * `ext_src[num_owned_cells][3]` - an array associating 3 external source
///      components with each (owned) cell in the domain
///
/// Active output fields:
///    * `cell[num_owned_cells][3]` - an array associating a 3-component source
///      value with each (owned) cell in the domain
///
/// Q-function context field labels:
///    * `time step` - the time step used by the operator
///    * `small h value` - the water height below which dry conditions are assumed
///    * `gravity` - the acceleration due to gravity [m/s/s]
///
/// @param [in]  config  RDycore's configuration
/// @param [in]  mesh    mesh defining the computational domain of the operator
/// @param [out] ceed_op a CeedOperator that is created and returned
/// @return 0 on success, or a non-zero error code on failure
static PetscErrorCode CreateCeedSource0Operator(const RDyConfig config, RDyMesh *mesh, CeedOperator *ceed_op) {
  PetscFunctionBeginUser;

  Ceed ceed = CeedContext();

  CeedInt num_sediment_comp = config.physics.sediment.num_classes;
  CeedInt num_flow_comp     = 3;  // NOTE: SWE assumed!
  CeedInt num_comp          = num_flow_comp + num_sediment_comp;

  RDyCells *cells = &mesh->cells;

  CeedQFunction qf;
  PetscCall(CreateSourceQFunction(ceed, config, &qf));

  // add inputs/outputs
  // NOTE: the order in which these inputs and outputs are specified determines
  // NOTE: their indexing within the Q-function's implementation
  CeedInt num_comp_geom = 2, num_comp_ext_src = num_comp;
  CeedInt num_mat_props = NUM_MATERIAL_PROPERTIES;
  PetscCallCEED(CeedQFunctionAddInput(qf, "geom", num_comp_geom, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "ext_src", num_comp_ext_src, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "mat_props", num_mat_props, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "riemannf", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddInput(qf, "q", num_comp, CEED_EVAL_NONE));
  PetscCallCEED(CeedQFunctionAddOutput(qf, "cell", num_comp, CEED_EVAL_NONE));

  // create vectors (and their supporting restrictions) for the operator
  CeedElemRestriction restrict_c, restrict_q, restrict_geom, restrict_ext_src, restrict_mat_props;
  CeedVector          geom, ext_src, mat_props;
  {
    PetscInt num_local_cells = mesh->num_cells;
    PetscInt num_owned_cells = mesh->num_owned_cells;

    // create a vector of geometric factors (elevation function derivatives)
    CeedScalar(*g)[num_comp_geom];
    CeedInt strides_geom[] = {num_comp_geom, 1, num_comp_geom};
    PetscCallCEED(
        CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_geom, num_owned_cells * num_comp_geom, strides_geom, &restrict_geom));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_geom, &geom, NULL));
    PetscCallCEED(CeedVectorSetValue(geom, 0.0));
    PetscCallCEED(CeedVectorGetArray(geom, CEED_MEM_HOST, (CeedScalar **)&g));
    for (CeedInt c = 0, owned_cell = 0; c < num_local_cells; ++c) {
      if (!cells->is_owned[c]) continue;
      g[owned_cell][0] = cells->dz_dx[c];
      g[owned_cell][1] = cells->dz_dy[c];
      ++owned_cell;
    }
    PetscCallCEED(CeedVectorRestoreArray(geom, (CeedScalar **)&g));

    // create a vector of external source terms
    CeedInt strides_ext_src[] = {num_comp_ext_src, 1, num_comp_ext_src};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_comp_ext_src, num_owned_cells * num_comp_ext_src, strides_ext_src,
                                                   &restrict_ext_src));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_ext_src, &ext_src, NULL));
    PetscCallCEED(CeedVectorSetValue(ext_src, 0.0));

    // create a vector that stores Manning's coefficient for the region of interest
    // NOTE: we zero-initialize this coefficient here; it must be set before use
    // NOTE: using (Get/Restore)OperatorMaterialProperty
    CeedInt strides_mat_props[] = {num_mat_props, 1, num_mat_props};
    PetscCallCEED(CeedElemRestrictionCreateStrided(ceed, num_owned_cells, 1, num_mat_props, num_owned_cells * num_mat_props, strides_mat_props,
                                                   &restrict_mat_props));
    PetscCallCEED(CeedElemRestrictionCreateVector(restrict_mat_props, &mat_props, NULL));
    PetscCallCEED(CeedVectorSetValue(mat_props, 0.0));

    // create element restrictions for (active) input/output cell states
    CeedInt *offset_c, *offset_q;
    PetscCall(PetscMalloc1(num_owned_cells, &offset_q));
    PetscCall(PetscMalloc1(num_owned_cells, &offset_c));
    for (CeedInt c = 0, owned_cell = 0; c < num_local_cells; ++c) {
      if (!cells->is_owned[c]) continue;
      offset_q[owned_cell] = c * num_comp;
      offset_c[owned_cell] = cells->local_to_owned[c] * num_comp;
      ++owned_cell;
    }
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_cells, 1, num_comp, 1, num_local_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            offset_q, &restrict_q));
    PetscCallCEED(CeedElemRestrictionCreate(ceed, num_owned_cells, 1, num_comp, 1, num_owned_cells * num_comp, CEED_MEM_HOST, CEED_COPY_VALUES,
                                            offset_c, &restrict_c));
    PetscCall(PetscFree(offset_c));
    PetscCall(PetscFree(offset_q));
    if (0) {
      PetscCallCEED(CeedElemRestrictionView(restrict_q, stdout));
      PetscCallCEED(CeedElemRestrictionView(restrict_c, stdout));
    }
  }

  // create the operator itself and assign its active/passive inputs/outputs
  PetscCallCEED(CeedOperatorCreate(ceed, qf, NULL, NULL, ceed_op));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "geom", restrict_geom, CEED_BASIS_NONE, geom));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "ext_src", restrict_ext_src, CEED_BASIS_NONE, ext_src));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "mat_props", restrict_mat_props, CEED_BASIS_NONE, mat_props));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "q", restrict_q, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));
  PetscCallCEED(CeedOperatorSetField(*ceed_op, "cell", restrict_c, CEED_BASIS_NONE, CEED_VECTOR_ACTIVE));

  // clean up
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_ext_src));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_geom));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_mat_props));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_c));
  PetscCallCEED(CeedElemRestrictionDestroy(&restrict_q));
  PetscCallCEED(CeedVectorDestroy(&geom));
  PetscCallCEED(CeedVectorDestroy(&ext_src));
  PetscCallCEED(CeedVectorDestroy(&mat_props));
  PetscCallCEED(CeedQFunctionDestroy(&qf));

  PetscFunctionReturn(CEED_ERROR_SUCCESS);
}

/// Creates a CEED source operator appropriate for the given configuration.
/// @param [in]    config              the configuration defining the physics and numerics for the new operator
/// @param [in]    mesh                a mesh containing geometric and topological information for the domain
/// @param [out]   source_op           the newly created operator
/// @return 0 on success, or a non-zero error code on failure
PetscErrorCode CreateCeedSourceOperator(RDyConfig *config, RDyMesh *mesh, CeedOperator *source_op) {
  PetscFunctionBegin;

  Ceed ceed = CeedContext();

  PetscCall(CeedCompositeOperatorCreate(ceed, source_op));

  CeedOperator source_0;
  PetscCall(CreateCeedSource0Operator(*config, mesh, &source_0));
  PetscCall(CeedCompositeOperatorAddSub(*source_op, source_0));

  PetscFunctionReturn(PETSC_SUCCESS);
}
#pragma GCC diagnostic   pop
#pragma clang diagnostic pop
