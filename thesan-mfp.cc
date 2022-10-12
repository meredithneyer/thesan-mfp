/*!
 * \file    thesan-mfp.cc
 *
 * \brief   Master file containing all program functions.
 *
 * \details This file contains the main function with high level parsing of
 *          command line arguments, parallelism, and program functionality.
 */

// Standard headers
#include <iomanip>    // int -> string
#include <iostream>   // input / output
#include <string>     // Standard strings
#include <vector>     // Standard vectors
#include <random>     // Random numbers
#include <sys/stat.h> // Make directories
#include <time.h>     // Timing utilities
#include <omp.h>      // OpenMP parallelism
#include <hdf5.h>     // HDF5 read / write

// Standard library usage
using std::cout;
using std::endl;
using std::string;
using std::to_string;
using std::vector;
typedef long long myint;

// Configuration constants
static const bool VERBOSE = true;            // Print extra information
static const bool USE_RENDERS = false;       // Use renders instead of smooth_renders
// static const int n_mfp = 10000;              // Number of mfp samplings (10^4)
// static const int n_mfp = 100000;             // Number of mfp samplings (10^5)
// static const int n_mfp = 1000000;            // Number of mfp samplings (10^6)
static const int n_mfp = 10000000;           // Number of mfp samplings (10^7)
// static const int n_mfp = 100000000;          // Number of mfp samplings (10^8)
// static const int n_mfp = 1000000000;         // Number of mfp samplings (10^9)
static const int n_bins_per_cell = 8;        // Number of histogram bins per cell
static const int n_loops_per_box = 2;        // Maximum distance in units of box size
static const double _2pi = 2. * M_PI;

// Distinguish between directions
enum DirectionTypes { PosX, NegX, PosY, NegY, PosZ, NegZ };

// Main control variables
static string executable;                    // Executable name (command line)
static string ren_dir;                       // Initial conditions directory (command line)
static string snap_str;                      // Snapshot number (option command line)
static int snap = -1;                        // Snapshot number (converted from string)
static string out_dir;                       // Output directory name

// OpenMP variables
static int thread;                           // OpenMP local thread
static int n_threads;                        // Number of OpenMP threads
static unsigned long long seed;              // Seed for random number generator
#pragma omp threadprivate(seed, thread)      // Unique to each thread

// Header variables
static double a, BoxSize, PixSize, InvPix, h, Omega0, OmegaBaryon, OmegaLambda;
static double UnitLength_in_cm, UnitMass_in_g, UnitVelocity_in_cm_per_s;
static double length_to_cgs, volume_to_cgs, mass_to_cgs, velocity_to_cgs;

// Grid variables
static int n_files = 1;                      // Number of render files
static myint Ngrid;                          // Number of pixels on a side
static myint Ngrid2;                         // Save hyperslab size
static myint Ngrid3;                         // Total number of cells
static double Ngrid_double;                  // Number of pixels on a side (double)
static double Ngrid3_double;                 // Total number of cells (double)
static double l_max;                         // Max ray-tracing distance (cell width units)
static int n_bins;                           // Number of histogram bins
static vector<double> edges;                 // Cell edge positions (cell width units)
static vector<double> centers;               // Cell center positions (cell width units)
static vector<float> HII_Fraction;           // HII fraction of each grid cell
static vector<int> mfp_hist;                 // Histogram of bubble sizes

static void read_header();                   // Read header information
static void read_data();                     // Read grid data
static void read_render_data();              // Read grid data
static void apply_threshold();               // Apply HII fraction threshold
static void calculate_mfps();                // Calculate all mean-free-paths
static double calculate_mfp();               // Calculate mean-free-path (cell width units)
static void write_data();                    // Write data to an output file

// Simple function to print an error message and exit
#define error(message) { error_info(message, __FILE__, __LINE__, __func__); }
template <typename T>
void error_info(const T message, const string file, const int line, const string func) {
  std::cerr << "\nError: " << message
            << "\nCheck: " << file << ":" << line << " in " << func << "()" << endl;
  exit(EXIT_FAILURE);
}


/*! \brief Uniform random number generator for everyday use.
 *
 *  This is the Ranq1 random number generator suggested by Numerical Recipes.
 *  The period is \f$\sim 1.8 \times 10^{19}\f$ sufficient for our applications.
 *
 *  \return Random number drawn uniformly in the interval \f$[0,1)\f$.
 */
double ran() {
  seed ^= seed >> 21;
  seed ^= seed << 35;
  seed ^= seed >> 4;
  return 5.42101086242752217E-20 * (seed * 2685821657736338717LL);
}


/*! \brief Main function of the program.
 *
 *  \param[in] argc Number of arguments passed to the program.
 *  \param[in] argv Full list of arguments passed to the program.
 *
 *  \return Exit code (0 = success, 1 = failure).
 */
int main(int argc, char** argv) {
  // Start the clock
  const clock_t start = clock();

  // Get the number of OpenMP threads
  #pragma omp parallel
  #pragma omp single
  n_threads = omp_get_num_threads();

  // Set unique thread and random seed
  #pragma omp parallel
  {
    thread = omp_get_thread_num();           // Get the thread number once
    std::random_device rd;                   // Obtains seeds for the random number engine
    seed = rd();                             // Set random seeds for parallel processes
  }

  // Configuration and general error checking
  executable = argv[0];                      // Executable name (command line)
  if (argc != 4)
    error("Please specify all arguments: " + executable + " [ren_dir] [out_dir] [snapshot]");
  ren_dir = argv[1];                         // Initial conditions directory
  out_dir = argv[2];                         // Output directory name
  snap_str = argv[3];                        // Snapshot number
  snap = std::stoi(snap_str);                // Convert to integer
  if (snap < 0)
    error("Expected snapshot number to be >= 0, but recieved " + snap_str);
  snap_str = "_" + string(3 - snap_str.length(), '0') + snap_str; // %03d format
  int status = mkdir(out_dir.c_str(), S_IRWXU | S_IRGRP | S_IXGRP);
  if (status != 0 && errno != EEXIST)
    error("Could not create the output directory: " + out_dir);

  // Read header information
  read_header();

  cout << " ___       ___  __             \n"
       << "  |  |__| |__  /__`  /\\  |\\ |\n"
       << "  |  |  | |___ .__/ /--\\ | \\|\n"
       << "\nInput Directory: " << ren_dir
       << "\nOutput Directory: " << out_dir
       << "\nRunning with " << n_threads << " threads"
       << "\n\nSnap " << snap << ": Ngrid = " << Ngrid
       << "\n\nz = " << 1./a - 1. << ", a = " << a << ", h = " << h
       << "\nBoxSize = " << 1e-3*BoxSize << " cMpc/h = " << 1e-3*BoxSize/h
       << " cMpc\nPixSize = " << 1e-3*PixSize << " cMpc/h = " << 1e-3*PixSize/h << " cMpc" << endl;

  // Read grid data
  if (USE_RENDERS)
    read_render_data();
  else
    read_data();

  // Apply HII fraction threshold
  apply_threshold();

  // Calculate all mean-free-paths
  calculate_mfps();

  // Write mfp data to a file
  write_data();

  // Free memory (overwrite with empty vectors)
  mfp_hist = vector<int>();
  HII_Fraction = vector<float>();
  centers = vector<double>();
  edges = vector<double>();

  // End the clock and report timings
  const clock_t end = clock();
  const double cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  cout << "\nElapsed time = " << cpu_time_used << " seconds" << endl;

  return 0;
}


//! \brief Read header information.
static void read_header() {
  string filename;
  if (USE_RENDERS)
    filename = ren_dir + "/render" + snap_str + "/render" + snap_str + ".000.hdf5";
  else
    filename = ren_dir + "/smooth" + snap_str + ".hdf5";

  hid_t file_id, group_id, attribute_id;
  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
  group_id = H5Gopen(file_id, "Header", H5P_DEFAULT);

  if (USE_RENDERS) {
    attribute_id = H5Aopen_name(group_id, "NumFiles");
    H5Aread(attribute_id, H5T_NATIVE_INT, &n_files);
    H5Aclose(attribute_id);
  }

  int NumPixels; // Saved as int type
  attribute_id = H5Aopen_name(group_id, "NumPixels");
  H5Aread(attribute_id, H5T_NATIVE_INT, &NumPixels);
  H5Aclose(attribute_id);
  Ngrid = NumPixels;                         // Convert to myint type
  Ngrid2 = Ngrid * Ngrid;                    // Number of grid cells in slab
  Ngrid3 = Ngrid * Ngrid2;                   // Total number of grid cells
  Ngrid_double = double(Ngrid);              // As double type
  Ngrid3_double = double(Ngrid3);            // As double type
  l_max = 0.999999999 * double(Ngrid) * double(n_loops_per_box); // Max distance (cell width units)
  edges.resize(Ngrid+1);                     // Cell edge positions (cell width units)
  centers.resize(Ngrid);                     // Cell center positions (cell width units)
  #pragma omp parallel for
  for (int i = 0; i <= Ngrid; ++i)
    edges[i] = double(i);                    // Grid is in cell width units
  #pragma omp parallel for
  for (int i = 0; i < Ngrid; ++i)
    centers[i] = 0.5 * (edges[i] + edges[i+1]); // Midpoint of edges

  attribute_id = H5Aopen_name(group_id, "Time");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &a);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "BoxSize");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &BoxSize);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "HubbleParam");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &h);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "Omega0");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &Omega0);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "OmegaBaryon");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &OmegaBaryon);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "OmegaLambda");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &OmegaLambda);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "UnitLength_in_cm");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &UnitLength_in_cm);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "UnitMass_in_g");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &UnitMass_in_g);
  H5Aclose(attribute_id);

  attribute_id = H5Aopen_name(group_id, "UnitVelocity_in_cm_per_s");
  H5Aread(attribute_id, H5T_NATIVE_DOUBLE, &UnitVelocity_in_cm_per_s);
  H5Aclose(attribute_id);

  H5Gclose(group_id);
  H5Fclose(file_id);

  PixSize = BoxSize / double(Ngrid);         // Pixel width
  InvPix = double(Ngrid) / BoxSize;          // Inverse pixel width
  length_to_cgs = a * UnitLength_in_cm / h;
  volume_to_cgs = length_to_cgs * length_to_cgs * length_to_cgs;
  mass_to_cgs = UnitMass_in_g / h;
  velocity_to_cgs = sqrt(a) * UnitVelocity_in_cm_per_s;
}


//! \brief Read grid data.
static void read_data() {
  string filename = ren_dir + "/smooth" + snap_str + ".hdf5";

  hid_t file_id, dataset;
  file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);

  HII_Fraction.resize(Ngrid3);               // Allocate space
  dataset = H5Dopen(file_id, "HII_Fraction", H5P_DEFAULT);
  H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, HII_Fraction.data());
  H5Dclose(dataset);

  H5Fclose(file_id);

  // Print data
  if (VERBOSE) {
    cout << "\nHII_Fraction = ["
         << HII_Fraction[0] << " "
         << HII_Fraction[1] << " "
         << HII_Fraction[2] << " ... "
         << HII_Fraction[Ngrid3-3] << " "
         << HII_Fraction[Ngrid3-2] << " "
         << HII_Fraction[Ngrid3-1] << "]" << endl;
  }
}


//! \brief Read render grid data.
static void read_render_data() {
  hid_t file_id, dataspace, dataset;

  HII_Fraction.resize(Ngrid3);               // Allocate space

  myint offset = 0;
  for (int i = 0; i < n_files; ++i) {
    string i_str = to_string(i);
    i_str = "." + string(3 - i_str.length(), '0') + i_str; // %03d format
    string filename = ren_dir + "/render" + snap_str + "/render" + snap_str + i_str+ "." + "hdf5";

    file_id = H5Fopen(filename.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT);
    dataset = H5Dopen(file_id, "HII_Fraction", H5P_DEFAULT);
    dataspace = H5Dget_space(dataset);
    const int n_dims = H5Sget_simple_extent_ndims(dataspace);
    hsize_t dims[n_dims];
    H5Sget_simple_extent_dims(dataspace, dims, NULL);
    H5Dread(dataset, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, &HII_Fraction[offset]);
    H5Dclose(dataset);
    H5Fclose(file_id);
    myint n_buf = dims[0];
    offset += n_buf;
  }

  // Print data
  if (VERBOSE) {
    cout << "\nHII_Fraction = ["
         << HII_Fraction[0] << " "
         << HII_Fraction[1] << " "
         << HII_Fraction[2] << " ... "
         << HII_Fraction[Ngrid3-3] << " "
         << HII_Fraction[Ngrid3-2] << " "
         << HII_Fraction[Ngrid3-1] << "]" << endl;
  }
}


//! \brief Apply HII fraction threshold.
static void apply_threshold() {
  // Set HII fraction to either 0 or 1
  myint n_neutral = 0;
  double x_ionized = 0.;
  #pragma omp parallel for reduction(+:n_neutral) reduction(+:x_ionized)
  for (myint i = 0; i < Ngrid3; ++i) {
    x_ionized += HII_Fraction[i];
    if (HII_Fraction[i] > 0.5) {
      HII_Fraction[i] = 1.;
    } else {
      HII_Fraction[i] = 0.;
      n_neutral++;
    }
  }

  // Print data
  if (VERBOSE) {
    cout << "\nHII_Fraction = ["
         << HII_Fraction[0] << " "
         << HII_Fraction[1] << " "
         << HII_Fraction[2] << " ... "
         << HII_Fraction[Ngrid3-3] << " "
         << HII_Fraction[Ngrid3-2] << " "
         << HII_Fraction[Ngrid3-1] << "]" << endl;
    cout << "\nGlobal x_HI = " << 1. - x_ionized / double(Ngrid3)
         << "  (x_HI_thresholds = " << double(n_neutral) / double(Ngrid3) << ")" << endl;
  }
}


//! \brief Calculate all mean-free-paths and bin into a histogram.
static void calculate_mfps() {
  n_bins = int(Ngrid) * n_bins_per_cell * n_loops_per_box; // Number of bins
  const double bpc = double(n_bins_per_cell); // Bins per cell (as a double)
  mfp_hist = vector<int>(n_bins);            // Allocate space and initialize to zero

  #pragma omp parallel for
  for (int i_mfp = 0; i_mfp < n_mfp; ++i_mfp) {
    const double mfp = calculate_mfp();      // Mean-free-path in units of cell widths
    const int i_bin = floor(mfp * bpc);      // Mean-free-path bin (oversampled units)
    if (i_bin < 0) error("i_bin < 0");
    if (i_bin >= n_bins) error("i_bin >= n_bins");
    #pragma omp atomic                       // Atomic is for thread safety
    mfp_hist[i_bin]++;                       // Increment histogram counter
  }

  // Print histogram
  if (VERBOSE) {
    cout << "\nmfp_hist = ["
         << mfp_hist[0] << " "
         << mfp_hist[1] << " "
         << mfp_hist[2] << " ... "
         << mfp_hist[n_bins-3] << " "
         << mfp_hist[n_bins-2] << " "
         << mfp_hist[n_bins-1] << "]" << endl;
  }
}


//! \brief Calculate random mean-free-path, returning in units of cell widths.
static double calculate_mfp() {
  // Select a random ionized cell
  myint ix, iy, iz, cell;                    // Cell indices
  do {
    cell = floor(ran() * Ngrid3_double);     // Random cell
  } while (HII_Fraction[cell] <= 0.5);       // Ensure cell is ionized

  // Initialize (x,y,z) position and indices
  myint prev_cell = cell;                    // Temporary cell indices
  ix = prev_cell / Ngrid2;                   // x index
  prev_cell -= ix * Ngrid2;                  // Hyperslab remainder
  iy = prev_cell / Ngrid;                    // y index
  iz = prev_cell - iy * Ngrid;               // Column remainder
  double x = centers[ix];                    // Cell center position
  double y = centers[iy];
  double z = centers[iz];
  double l_tot = 0.;                         // Cumulative distance (cell width units)
  double l, l_comp;                          // Comparison distances
  int direction_type;                        // Direction enumerated type

  // Select a random direction (isotropic distribution)
  const double theta = acos(2. * ran() - 1.);
  const double sin_theta = sin(theta);
  const double phi = _2pi * ran();
  double kx = sin_theta * cos(phi);
  double ky = sin_theta * sin(phi);
  double kz = cos(theta);
  const double inv_norm = 1. / sqrt(kx*kx + ky*ky + kz*kz);
  kx *= inv_norm;                            // Normalize to avoid error
  ky *= inv_norm;
  kz *= inv_norm;

  // Ray-tracing
  do {
    l = l_max;                               // Initialize to large value
    // x direction
    if (kx != 0.) {
      if (kx > 0.) {
        l_comp = (edges[ix+1] - x) / kx;     // Right x face
        if (l_comp < l) {
          l = l_comp;
          direction_type = PosX;
        }
      } else {
        l_comp = (edges[ix] - x) / kx;       // Left x face
        if (l_comp < l) {
          l = l_comp;
          direction_type = NegX;
        }
      }
    }
    // y direction
    if (ky != 0.) {
      if (ky > 0.) {
        l_comp = (edges[iy+1] - y) / ky;     // Right y face
        if (l_comp < l) {
          l = l_comp;
          direction_type = PosY;
        }
      } else {
        l_comp = (edges[iy] - y) / ky;       // Left y face
        if (l_comp < l) {
          l = l_comp;
          direction_type = NegY;
        }
      }
    }
    // z direction
    if (kz != 0.) {
      if (kz > 0.) {
        l_comp = (edges[iz+1] - z) / kz;     // Right z face
        if (l_comp < l) {
          l = l_comp;
          direction_type = PosZ;
        }
      } else {
        l_comp = (edges[iz] - z) / kz;       // Left z face
        if (l_comp < l) {
          l = l_comp;
          direction_type = NegZ;
        }
      }
    }

    // Update position and indices based on the direction
    switch (direction_type) {                // Actions based on direction
      case PosX:                             // Positive X
        ++ix;
        if (ix == Ngrid) {
          ix = 0;                            // Periodic boundaries
          x -= Ngrid_double;
        }
        break;
      case NegX:                             // Negative X
        --ix;
        if (ix == -1) {
          ix = Ngrid - 1;                    // Periodic boundaries
          x += Ngrid_double;
        }
        break;
      case PosY:                             // Positive Y
        ++iy;
        if (iy == Ngrid) {
          iy = 0;                            // Periodic boundaries
          y -= Ngrid_double;
        }
        break;
      case NegY:                             // Negative Y
        --iy;
        if (iy == -1) {
          iy = Ngrid - 1;                    // Periodic boundaries
          y += Ngrid_double;
        }
        break;
      case PosZ:                             // Positive Z
        ++iz;
        if (iz == Ngrid) {
          iz = 0;                            // Periodic boundaries
          z -= Ngrid_double;
        }
        break;
      case NegZ:                             // Negative Z
        --iz;
        if (iz == -1) {
          iz = Ngrid - 1;                    // Periodic boundaries
          z += Ngrid_double;
        }
        break;
      default:
        error("Unrecognized direction type in ray-tracing calculation.");
    }

    prev_cell = cell;                        // Reset previous cell index
    cell = (ix * Ngrid + iy) * Ngrid + iz;   // Update cell index
    if (l < 0.)
      l = 0.;                                // Avoid negative distances
    l_tot += l;                              // Add current path length
    x += l * kx; y += l * ky; z += l * kz;
  } while (HII_Fraction[cell] > 0.5 && l_tot < l_max); // Limit total distance by l_max

  if (l_tot > l_max)
    l_tot = l_max;                           // Adjust overshooting distance

  return l_tot;                              // Final mean-free-path (cell width units)
}


#define WRITE_ATTRIBUTE(attr_name, attr_value, attr_type)                                           \
  dataspace_id = H5Screate(H5S_SCALAR);                                                             \
  attribute_id = H5Acreate(group_id, attr_name, attr_type, dataspace_id, H5P_DEFAULT, H5P_DEFAULT); \
  status       = H5Awrite(attribute_id, attr_type, &(attr_value));                                  \
  status       = H5Aclose(attribute_id);                                                            \
  status       = H5Sclose(dataspace_id);

struct UnitAttrs
{
  double a;  // exponent of the cosmological a factor
  double h;  // exponent of the hubble parameter
  double L;  // length unit scaling
  double M;  // mass unit scaling
  double V;  // velocity unit scaling
  double c;  // conversion factor to cgs units (zero indicates dimensionless quantity, integer count, etc)
};

/*! \brief Function for setting units of an output field.
 *
 *  \param[in/out] *ua UnitAttrs pointer to be set.
 *  \param[in] a the exponent of the cosmological a factor.
 *  \param[in] h the exponent of the hubble parameter.
 *  \param[in] L the length unit scaling.
 *  \param[in] M the mass unit scaling.
 *  \param[in] V the velocity unit scaling.
 *  \param[in] c conversion factor to cgs units (zero indicates dimensionless
 *             quantity, integer count, etc).
 *
 *  \return void
 */
static inline void set_unit_attrs(struct UnitAttrs *ua, double a, double h, double L, double M, double V, double c)
{
  ua->a = a;
  ua->h = h;
  ua->L = L;
  ua->M = M;
  ua->V = V;
  ua->c = c;
}

/*! \brief Function for adding units to an output field.
 *
 *  \param[in] file_id specifies the file location.
 *  \param[in] name specifies the dataset location relative to file_id.
 *  \param[in] ua the UnitAttrs struct holding (a,h,L,M,V,c) attributes.
 *
 *  \return void
 */
static void write_units(hid_t file_id, const char *name, struct UnitAttrs *ua)
{
  herr_t status;
  hid_t dataspace_id, attribute_id;

  hid_t group_id = H5Dopen(file_id, name, H5P_DEFAULT); // group_id is for convenience (macro)

  WRITE_ATTRIBUTE("a_scaling", ua->a, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("h_scaling", ua->h, H5T_NATIVE_DOUBLE)

  WRITE_ATTRIBUTE("length_scaling", ua->L, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("mass_scaling", ua->M, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("velocity_scaling", ua->V, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("to_cgs", ua->c, H5T_NATIVE_DOUBLE)

  H5Dclose(group_id); // Close the dataset
}

//! \brief Writes out a vector quantity, e.g. (a1,a2,a3,...)
static void write_1d(hid_t file_id, vector<int>& vec, const char *name) {
  // Identifier
  hsize_t vec_size = vec.size();
  hid_t dataspace_id, dataset_id;
  hsize_t dims1d[1] = {vec_size};

  dataspace_id = H5Screate_simple(1, dims1d, NULL);
  dataset_id = H5Dcreate(file_id, name, H5T_NATIVE_INT, dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  H5Dclose(dataset_id);
  H5Sclose(dataspace_id);

  // Open dataset and get dataspace
  hid_t dataset   = H5Dopen(file_id, name, H5P_DEFAULT);
  hid_t filespace = H5Dget_space(dataset);

  // File hyperslab
  hsize_t file_offset[1] = {0};
  hsize_t file_count[1] = {vec_size};

  H5Sselect_hyperslab(filespace, H5S_SELECT_SET, file_offset, NULL, file_count, NULL);

  // Memory hyperslab
  hsize_t mem_offset[1] = {0};
  hsize_t mem_count[1] = {vec_size};

  hid_t memspace = H5Screate_simple(1, mem_count, NULL);
  H5Sselect_hyperslab(memspace, H5S_SELECT_SET, mem_offset, NULL, mem_count, NULL);

  // Write
  H5Dwrite(dataset, H5T_NATIVE_INT, memspace, filespace, H5P_DEFAULT, vec.data());

  // Close handles
  H5Sclose(memspace);
  H5Sclose(filespace);
  H5Dclose(dataset);
}


//! \brief Write data to an output file.
static void write_data() {
  string filename = out_dir + "/mfp" + snap_str + ".hdf5";

  // Identifiers
  herr_t status;
  hid_t file_id, group_id, dataspace_id, dataset_id, attribute_id;

  // Open file and write header
  file_id = H5Fcreate(filename.c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
  group_id = H5Gcreate(file_id, "Header", H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);

  // BoxSize, Time, Units, HubbleParam, Omega0, Redshift
  const double Redshift = 1.0 / a - 1.0;
  const int NumPixels = Ngrid;
  WRITE_ATTRIBUTE("BoxSize", BoxSize, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("NumPixels", NumPixels, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("NumBins", n_bins, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("NumSamples", n_mfp, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("NumBinsPerCell", n_bins_per_cell, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("NumLoopsPerBox", n_loops_per_box, H5T_NATIVE_INT)
  WRITE_ATTRIBUTE("Time", a, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("UnitLength_in_cm", UnitLength_in_cm, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("UnitMass_in_g", UnitMass_in_g, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("UnitVelocity_in_cm_per_s", UnitVelocity_in_cm_per_s, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("HubbleParam", h, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("Omega0", Omega0, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("OmegaBaryon", OmegaBaryon, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("OmegaLambda", OmegaLambda, H5T_NATIVE_DOUBLE)
  WRITE_ATTRIBUTE("Redshift", Redshift, H5T_NATIVE_DOUBLE)

  status = H5Gclose(group_id);

  // Smoothed regrid data
  write_1d(file_id, mfp_hist, "mfp_hist");

  // Close file
  H5Fclose(file_id);
}
