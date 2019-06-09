// Adapted from Water2D, THU Zhaowei Liu
#include "fvCFD.H"
#include "pimpleControl.H"
#include "singlePhaseTransportModel.H"
#include "turbulenceModel.H"
#include "wallFvPatch.H"
#include "emptyFvPatch.H"
#include "water2D.H"
#include "char.H"
#include "messageStream.H"

const scalar pi = 3.14159265;

    Info<< "Reading velocity field\n" << endl;
    volVectorField U
    (
        IOobject
        (
            "U",    // reading file name
            runTime.timeName(),  // time step
            mesh,  // registry
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE  // U.write()
        ),
        mesh
    );

    Info<< "Reading water surface evaluation field\n" << endl;
    volScalarField zeta
    (
        IOobject
        (
            "zeta",
            runTime.timeName(),
            mesh,
            IOobject::MUST_READ,
            IOobject::AUTO_WRITE
        ),
        mesh
    );   //  water surface level above reference surface

    Info<< "Reading bed evaluation field\n" << endl;
    volScalarField d
    (
        IOobject
        (
            "depth",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        ),
        mesh
    );   //  bed elevation under reference surface

    volScalarField h
    (
        IOobject
        (
             "h",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::AUTO_WRITE
        ),
        zeta + d,
        "zeroGradient"
    );

    Info<< "Reading roughness of cells\n" << endl;
    volScalarField n
    (
         IOobject
         (
               "n",
                runTime.constant(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
          ),
          mesh
    );  // roughness

    Info<< "Reading runoff coefficient of cells\n" << endl;
    volScalarField alpha
    (
         IOobject
         (
               "alpha",
                runTime.constant(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
          ),
          mesh
    );

    volScalarField gully_num
    (
            IOobject
            (
                "gully_num",
                runTime.constant(),
                mesh,
                IOobject::MUST_READ,
                IOobject::NO_WRITE
            ),
            mesh
        );   //  the number of gullies on a cell

    volScalarField conduit_level
    (
            IOobject
            (
                "conduit_level",
                runTime.timeName(),
                mesh,
                IOobject::READ_IF_PRESENT,
                IOobject::NO_WRITE
            ),
            mesh,
            dimensionedScalar("conduit_level", dimLength, 0.0)
        );   // the water level of the node in the conduit

    volScalarField q
    (
        IOobject
        (
            "q",
            runTime.timeName(),
            mesh,
            IOobject::READ_IF_PRESENT,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("q", dimLength/dimTime, 0.0)
    );  // the interaction flow between surface and conduit

    const scalarField& cellVolumes = mesh.V();

    IOdictionary transportProperties
    (
        IOobject
        (
            "transportProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    dimensionedScalar nu
    (
        transportProperties.lookup("nu")
    );  //  (viscousity ecofficient)

    IOdictionary environmentalProperties
    (
        IOobject
        (
            "environmentalProperties",
            runTime.constant(),
            mesh,
            IOobject::MUST_READ,
            IOobject::NO_WRITE
        )
    );

    dimensionedScalar ks
    (
        environmentalProperties.lookup("ks")
    );  //

    dimensionedScalar controlHeight
    (
        environmentalProperties.lookup("controlHeight")
    );

     dimensionedScalar flow_aug
     (
         environmentalProperties.lookup("flow_aug")
     );

    dimensionedScalar latitude
    (
        environmentalProperties.lookup("latitude")
    );

    const dimensionedScalar g
    (
        environmentalProperties.lookup("g")
    );

    dimensionedVector f
    (
        "f",
        dimless/dimTime,
        vector::zero
    );
    f.value().z() = 2.0 * (pi / (23.93444 * 1800)) * Foam::sin(latitude.value() / 180 * pi);

    dimensionedScalar Fr
    (
        "Fr",
        dimless,
        1.0
    );

    volScalarField Cf
    (
        IOobject
        (
             "Cf",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("Cf", dimless, 0.001),
        "zeroGradient"
    );

    volScalarField rCs
    (
        IOobject
        (
             "rCs",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        Foam::sqrt(Cf),
        "zeroGradient"
    );	 // reciprocal of the Chezy coeffcient

    volScalarField fricU
    (
        IOobject
        (
             "fricU",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        rCs * mag(U),
        "zeroGradient"
    );  // friction velocity

    volScalarField cZDdt
    (
        IOobject
        (
             "cZDdt",
            runTime.timeName(),
            mesh,
            IOobject::NO_READ,
            IOobject::NO_WRITE
        ),
        mesh,
        dimensionedScalar("cZDdt", dimless, 1.0),
        "zeroGradient"
    );

#   include "addSources.H"
#   include "computeWaterHeight.H"

    Info<< "Creating field hU\n" << endl;
    volVectorField hU
    (
        IOobject
        (
            "hU",
            runTime.timeName(),
            mesh
        ),
        mesh,
        U.dimensions()*h.dimensions(),
        vectorField(),
        volVectorField::GeometricBoundaryField(U.boundaryField())
    );
    hU = h * U;

#   include "createPhi.H"


    singlePhaseTransportModel laminarTransport(U, phi);
    autoPtr<turbulenceModel> turbulence
    (
        turbulenceModel::New(h, U, phi, laminarTransport)
    );
