#include <stdlib.h>
#include <fstream>
#include <string>
#include <cstdlib>
#include <math.h>
#include <sstream>
#include "fvCFD.H"
#include "pimpleControl.H"
#include "singlePhaseTransportModel.H"
#include "turbulenceModel.H"
#include "waterQualityModel.H"
#include "wallFvPatch.H"
#include "emptyFvPatch.H"
#include "volumeFluxInletVelocityFvPatchVectorField.H"
#include "OFstream.H"
#include "OFstream.C"
#include "regIOobjectWrite.C"
#include "water2D.H"
#include "swmm5.h"
#include <dlfcn.h>

struct Source{
    friend Istream& operator>>(Istream& is, Source& xyd);
    friend Ostream& operator<<(Ostream& os, const Source& xyd);

    label position;
    double value;
};

struct Gate{
    friend Istream& operator>>(Istream& is, Gate& xyd);
    friend Ostream& operator<<(Ostream& os, const Gate& xyd);

    label top;
    label bottom;
    double opening;
    double width;
};

struct Weir{
    friend Istream& operator>>(Istream& is, Weir& xyd);
    friend Ostream& operator<<(Ostream& os, const Weir& xyd);
    label top;
    label bottom;
    double elevation;
    double width;
};

// friend function for Source, Gate, Weir
Istream& operator>>(Istream& is, Source& xyd){
    token firstToken(is);

    if (firstToken.pToken() != token::BEGIN_LIST)
    {
        FatalIOErrorIn("operator>>(Istream&, List<T>&)", is)
            << "incorrect first token, expected '(' or '{', found "
            << firstToken.info()
            << exit(FatalIOError);
    }

    is >> xyd.position >> xyd.value;

    token lastToken(is);
    while(!(lastToken.pToken() == token::END_LIST))
    {
        is >> lastToken;
    }
    return is;
}

Ostream& operator<<(Ostream& os, const Source& xyd){
    os << '(' << xyd.position << ' ' << xyd.value << ')';
    return os;
}

Istream& operator>>(Istream& is, Gate& xyd){
    token firstToken(is);

    if (firstToken.pToken() != token::BEGIN_LIST)
    {
        FatalIOErrorIn("operator>>(Istream&, List<T>&)", is)
            << "incorrect first token, expected '(' or '{', found "
            << firstToken.info()
            << exit(FatalIOError);
    }

    is >> xyd.top >> xyd.bottom >> xyd.opening >> xyd.width;

    token lastToken(is);

    while(!(lastToken.pToken() == token::END_LIST))
    {
        is >> lastToken;
    }
    return is;
}

Ostream& operator<<(Ostream& os, const Gate& xyd){
    os << '(' << xyd.top << ' ' << xyd.bottom << ' ' << xyd.opening << ' ' << xyd.width << ')';
    return os;
}

Istream& operator>>(Istream& is, Weir& xyd){
    token firstToken(is);

    if (firstToken.pToken() != token::BEGIN_LIST)
    {
        FatalIOErrorIn("operator>>(Istream&, List<T>&)", is)
            << "incorrect first token, expected '(' or '{', found "
            << firstToken.info()
            << exit(FatalIOError);
    }

    is >> xyd.top >> xyd.bottom >> xyd.elevation >> xyd.width;

    token lastToken(is);

    while(!(lastToken.pToken() == token::END_LIST))
    {
        is >> lastToken;
    }
    return is;
}

Ostream& operator<<(Ostream& os, const Weir& xyd){
    os << '(' << xyd.top << ' ' << xyd.bottom << ' ' << xyd.elevation << ' ' << xyd.width << ')';
    return os;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
int main(int argc, char *argv[])
{
    argList::validArgs.append("Rainfile");
    argList::validArgs.append("SWMM_input");
    argList::validArgs.append("SWMM_report");
    argList::validArgs.append("SWMM_output");
    # include "setRootCase.H"
    # include "createTime.H"
    # include "createMesh.H"
    # include "createFields.H"  // Time created first, Mesh created next, Field created finally

    # include "computeWaterHeight.H"
    # include "computeFrictionVelocity.H"
    # include "readTimeControls.H"
    pimpleControl pimple(mesh);
    // get the handle of SWMM shared library/dynamic link library
    void *handle = dlopen("/root/OpenFOAM/root-2.3.0/platforms/linux64GccDPOpt/lib/libswmm5.so", RTLD_LAZY);
    if (!handle){
        Info << "\n Open DLL error\n" << endl;
        Info << dlerror() << endl;
        return -1;
    }

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //
    // get the functions in SWMM share library
    typedef int (*swmm_open_t)(char* f1, char *f2, char* f3);
    typedef int (*swmm_start_t)(int saveResults);
    typedef int (*swmm_step_t)(double* elapsedTime);
    typedef int (*swmm_end_t)(void);
    typedef int (*swmm_report_t)();
    typedef int (*swmm_close_t)();
    typedef int (*Swmm_link_t)(double* SWMM_DT);
    typedef int (*Swmm_valid_t)(int NodeID, double* outflow);
    typedef int (*Swmm_to_2D_t)(int* NodeID, double* Node_h,double* Cell_Q, double flow_aug);
    typedef int (*twoD_to_Swmm_t)(int* NodeID_j, double* Node_h, double flow_aug);

    swmm_open_t swmm_open = (swmm_open_t) dlsym(handle, "swmm_open");
    swmm_start_t swmm_start = (swmm_start_t)dlsym(handle, "swmm_start");
    swmm_step_t swmm_step = (swmm_step_t)dlsym(handle, "swmm_step");
    swmm_end_t swmm_end = (swmm_end_t)dlsym(handle, "swmm_end");
    swmm_report_t swmm_report = (swmm_report_t)dlsym(handle, "swmm_report");
    swmm_close_t swmm_close = (swmm_close_t)dlsym(handle, "swmm_close");
    Swmm_link_t Swmm_link = (Swmm_link_t)dlsym(handle, "Swmm_Link");
    Swmm_valid_t Swmm_valid = (Swmm_valid_t)dlsym(handle, "Swmm_valid");
    Swmm_to_2D_t Swmm_to_2D = (Swmm_to_2D_t)dlsym(handle, "Swmm_to_2D");
    twoD_to_Swmm_t twoD_to_Swmm = (twoD_to_Swmm_t)dlsym(handle, "twoD_to_Swmm");

    Info << "\n SWMM funcs are loaded successfully\n" << endl;
    // initialize Water2D-SWMM settings, e.g. coeff, time step
    double flow_aug_coef = flow_aug.value();
    long newHour, oldHour = 0;
    long theDay, theHour;
    int minute = 0;
    int line_index = 0;
    char* arg_data = (char *)malloc((args[1].length()+1)*sizeof(char));
    args[1].copy(arg_data, args[1].length(), 0);
    // read rain file
    std::ifstream rain_file(arg_data);
    free(arg_data);
    std::string line;
    double rain;
    // result of outflow for validation
    OFstream valid_file("validation.txt");

    // initialize SWMM flags
    static int IsOpenFlag = false;
    static int IsStartedFlag = false;
    static int SaveResultsFlag = false;

    // open SWMM files & read input data
    int ErrorCode = 0;
    char* arg_data_2 = (char *)malloc((args[2].length()+1)*sizeof(char));
    args[2].copy(arg_data_2, args[2].length(), 0);
    char* arg_data_3 = (char *)malloc((args[3].length()+1)*sizeof(char));
    args[3].copy(arg_data_3, args[3].length(), 0);
    char* arg_data_4 = (char *)malloc((args[4].length()+1)*sizeof(char));
    args[4].copy(arg_data_4, args[4].length(), 0);
    swmm_open(arg_data_2, arg_data_3, arg_data_4);
    free(arg_data_2);
    free(arg_data_3);
    free(arg_data_4);

    if (!ErrorCode){
        Info << "\n SWMM project opening\n" << endl;
        swmm_start(true);
        // SWMM simulation time, also synchronization time, in decimal days (SWMM simulation requirement)
        double elapsedTime = 0.0;
        double deltaT_2D = runTime.deltaTValue();
        // deltaT for SWMM simulation, in seconds
        double deltaT_SWMM = 0.0;
        // get SWMM fixed routing step
        Swmm_link(&deltaT_SWMM);
        Info << "The SWMM routing step is: " << deltaT_SWMM << "s\n" << endl;
        Info << "The 2D simulation step is: " << deltaT_2D << "s\n" << endl;
        // SWMM deltaT / 2D deltaT, synchronize every 30*2D deltaT
        int block_sync = deltaT_SWMM/deltaT_2D;
        int block_min = 60/deltaT_SWMM;
        int block = block_sync;
        int block_write = 60;
        if (!ErrorCode){
          do{
            Info<< "\nStarting time loop\n" << endl;
            Info << "\n SWMM Simulation time = " << runTime.timeName() << nl << endl;
            // read net rainfall
            Info<< "Update cell's q from new rainfall\n" << endl;
            while(std::getline(rain_file, line)){
                if (line_index == minute){
                    std::istringstream  line_stream(line);
                    line_stream >> rain;
                    forAll(cellVolumes, i){
                        q[i] += rain*alpha[i];
                        break;
                    }
                }
                else{
                    line_index++;
                }
            }
            line_index = 0; // reset index of lines to 0
            block_min--;
            if (block_min<=0){
                 minute++;
                 block_min=60/deltaT_SWMM;
            }
            // execute 2D simulation for 60s
            while (runTime.run() && block>0 ){
                Info<< "Water2D simulation time = " << runTime.timeName() << nl << endl;
                #include "readTimeControls.H"
                #include "CourantNo.H"
                #include "setDeltaT.H"
                // --- Pressure-velocity PIMPLE corrector loop start
                while (pimple.loop()){   // Pimple = PISO + simple
                 // Momentum predictor
                    phiv = phi/fvc::interpolate(h/cZDdt);   // phi: mass flux
                    fvVectorMatrix hUEqn
                    (
                        fvm::ddt(hU)
                    + fvm::div(phiv, hU)  // for diffusive wave simulation, e.g. dry and wet condition, just drop this term
                    + turbulence->divR(hU)
                    );

                    hUEqn.relax();

                    if (pimple.momentumPredictor()){
                        solve(hUEqn == (-g * h * fvc::grad(zeta) - fvm::Sp(rCs * fricU / h, hU))) ;
                        #include "computeFrictionVelocity.H"
                    }

                    // --- Pressure corrector loop
                    while (pimple.correct()){
                        volScalarField rUA = 1.0 / (hUEqn.A() + rCs * fricU / h);   // hUEqn.A() return the central coefficient of hUEqn
                        volScalarField ghrUAf = g * rUA * h;

                        hU = rUA * hUEqn.H();  // hUEqn.H() return the (L+U)\cdot u_0
			
                        phi = (fvc::interpolate(hU) & mesh.Sf())
                            + fvc::interpolate(rUA)*fvc::ddtCorr(h, hU, phi);

                        while (pimple.correctNonOrthogonal()){
                          // water level corrector
                            volScalarField a = cZDdt;

                            fvScalarMatrix zetaEqn(
                                fvm::ddt(a, zeta)  // use zeta to compute h
                                +   fvc::div(phi)
                                -   fvm::laplacian(ghrUAf, zeta)
                                -   q   // add source term here?
                            );
                            zetaEqn.solve(mesh.solver(zeta.select(pimple.finalInnerIter())));
		
                            if (pimple.finalNonOrthogonalIter()){
                                phi += zetaEqn.flux();
                            }
                        }
				
                    #include "caculateGateFlux.H"
                    #include "caculateWeirFlux.H"

                    hU -= ghrUAf * fvc::grad(zeta);
                    hU.correctBoundaryConditions();
                    U.correctBoundaryConditions();
                    U = hU / h;
                    phi0 = phi;
                    #include "computeWaterHeight.H"
                    #include "computeFrictionVelocity.H"
              }
            }  //pimple loop end
	
            turbulence->correct();
            //waterQuality->correct();

            #include "computeFlux.H"
            Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
                << "  ClockTime = " << runTime.elapsedClockTime() << " s"
                << nl << endl;
            runTime++;
            block--;
            block_write--;
            if(block_write<=0){
                runTime.writeNow();  // write 2D simulation result
                block_write = 60;
            }
            }  // end 2D simulation
            // external inflow to SWMM according to water2D depth on surface & old depth in conduits
           Info<< "2D to SWMM\n" << endl;
           forAll(cellVolumes,i){
           int n = gully_num[i];
                if (n != -1){
                //Info<< h[i] <<"\n" << endl;
                   twoD_to_Swmm(&n, &h[i], flow_aug_coef);
                }
            }
        // execute SWMM for one step computation
        swmm_step(&elapsedTime);
        newHour = (long)(elapsedTime*24.0);
        if (newHour > oldHour){
            theDay = (long) elapsedTime;
            theHour = (long)((elapsedTime - floor(elapsedTime))*24);
            Info<< "the day: " << theDay << "the hour: " << theHour << "\n" << endl;
            oldHour = newHour;
        }
        Info<< "the elapsed time: " << elapsedTime << "\n" << endl;
        Info<< "SWMM steps successfully\n" << endl;
        // select junctions of conduits'outflow for Water2D-SWMM validation
        double outflow = 0;
        Swmm_valid(32, &outflow);
        valid_file<< "R32 " << outflow << "\t";
        Swmm_valid(95, &outflow);
        valid_file<< "UJ37 " << outflow << "\t";
        Swmm_valid(16, &outflow);
        valid_file<< "R16 " << outflow << "\t";
        Swmm_valid(112, &outflow);
        valid_file<< "UJ54 " << outflow << "\t";
        Swmm_valid(90, &outflow);
        valid_file<< "UJ32 " << outflow << "\t";
        Swmm_valid(328, &outflow);
        valid_file<< "Y72 " << outflow << "\t";
        Swmm_valid(345, &outflow);
        valid_file<< "Y89 " << outflow << "\n";
        // update water2D source term
        Info<< "Update cell's q from new water level of conduits\n" << endl;
        forAll(cellVolumes,i){
             // First, reset flow to 0 for next Water2D-SWMM simualtion
             q[i] = 0;
              // Add flow from SWMM
             int n = gully_num[i];
             double q_cell = 0;
             if (n == -1){
                continue;
             }
             else{
                 Swmm_to_2D(&n, &h[i], &q_cell, flow_aug_coef);
                 q[i] += q_cell/cellVolumes[i];
             }
         }
         block = block_sync;
       }
       while(elapsedTime > 0.0 && !ErrorCode);
       Info<< "End\n" << endl;
       swmm_end();
       swmm_report();
       swmm_close();
       }//SWMM simulation end
       else{
           Info<< "SWMM can't be started\n" << endl;
       }
    }
    else{
        Info<< "SWMM can't be opened\n" << endl;
    }
    return 0;
}

// ************************************************************************* //
