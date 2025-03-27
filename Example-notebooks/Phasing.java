/* Copyright 2002-2023 CS GROUP
 * Licensed to CS GROUP (CS) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * CS licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.orekit.tutorials.bodies;

import java.io.File;
import java.io.IOException;
import java.io.PrintStream;
import java.net.URISyntaxException;
import java.net.URL;
import java.nio.charset.StandardCharsets;
import java.time.LocalDateTime;
import java.time.ZoneId;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.stream.Collectors;

import org.hipparchus.analysis.UnivariateFunction;
import org.hipparchus.analysis.solvers.BaseUnivariateSolver;
import org.hipparchus.analysis.solvers.BracketingNthOrderBrentSolver;
import org.hipparchus.analysis.solvers.UnivariateSolverUtils;
import org.hipparchus.exception.LocalizedCoreFormats;
import org.hipparchus.exception.MathRuntimeException;
import org.hipparchus.geometry.euclidean.threed.Vector3D;
import org.hipparchus.ode.nonstiff.DormandPrince853Integrator;
import org.hipparchus.optim.nonlinear.vector.leastsquares.LeastSquaresBuilder;
import org.hipparchus.optim.nonlinear.vector.leastsquares.LeastSquaresOptimizer;
import org.hipparchus.optim.nonlinear.vector.leastsquares.LeastSquaresProblem;
import org.hipparchus.optim.nonlinear.vector.leastsquares.LevenbergMarquardtOptimizer;
import org.hipparchus.util.FastMath;
import org.hipparchus.util.MathUtils;
import org.hipparchus.util.SinCos;
import org.orekit.bodies.BodyShape;
import org.orekit.bodies.CelestialBodyFactory;
import org.orekit.bodies.GeodeticPoint;
import org.orekit.data.DataContext;
import org.orekit.data.DataProvidersManager;
import org.orekit.data.DirectoryCrawler;
import org.orekit.errors.OrekitException;
import org.orekit.files.ccsds.definitions.BodyFacade;
import org.orekit.files.ccsds.definitions.FrameFacade;
import org.orekit.files.ccsds.definitions.TimeSystem;
import org.orekit.files.ccsds.ndm.WriterBuilder;
import org.orekit.files.ccsds.ndm.odm.OdmHeader;
import org.orekit.files.ccsds.ndm.odm.oem.OemMetadata;
import org.orekit.files.ccsds.ndm.odm.oem.OemWriter;
import org.orekit.files.ccsds.ndm.odm.oem.StreamingOemWriter;
import org.orekit.files.ccsds.utils.generation.Generator;
import org.orekit.files.ccsds.utils.generation.KvnGenerator;
import org.orekit.forces.gravity.HolmesFeatherstoneAttractionModel;
import org.orekit.forces.gravity.ThirdBodyAttraction;
import org.orekit.forces.gravity.potential.GravityFieldFactory;
import org.orekit.forces.gravity.potential.NormalizedSphericalHarmonicsProvider;
import org.orekit.frames.Frame;
import org.orekit.frames.FramesFactory;
import org.orekit.models.earth.ReferenceEllipsoid;
import org.orekit.orbits.CircularOrbit;
import org.orekit.orbits.Orbit;
import org.orekit.orbits.OrbitType;
import org.orekit.orbits.PositionAngleType;
import org.orekit.propagation.Propagator;
import org.orekit.propagation.SpacecraftState;
import org.orekit.propagation.numerical.NumericalPropagator;
import org.orekit.time.AbsoluteDate;
import org.orekit.time.TimeComponents;
import org.orekit.time.TimeScalarFunction;
import org.orekit.time.TimeScale;
import org.orekit.time.TimeScalesFactory;
import org.orekit.tutorials.yaml.TutorialForceModel.TutorialGravity;
import org.orekit.tutorials.yaml.TutorialOrbit;
import org.orekit.utils.Constants;
import org.orekit.utils.IERSConventions;
import org.orekit.utils.PVCoordinates;
import org.orekit.utils.SecularAndHarmonic;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.dataformat.yaml.YAMLFactory;

/** Orekit tutorial for setting up a Sun-synchronous Earth-phased Low Earth Orbit.
 * @author Luc Maisonobe
 */
public class Phasing {

    /** Key to print files generation. */
    private static final String GENERATED = "%n%s generated%n";

    /** Key to print initial osculating orbit. */
    private static final String INITIAL_OSC_ORBIT = "initial osculating orbit%n";

    /** Key to print final osculating orbit. */
    private static final String FINAL_OSC_ORBIT = "%nfinal osculating orbit%n";

    /** Key to print initial and final epochs. */
    private static final String EPOCH = "    date = %s%n";

    /** Key to print initial and final orbital elements. */
    private static final String ORBITAL_ELEMENTS = "    a = %14.6f m, ex = %17.10e, ey = %17.10e, i = %12.9f deg, Ω = %12.9f deg%n";

    /** Key to pthe iteration report. */
    private static final String ITERATION_REPORT = " iteration %2d: Δa = %12.6f m, Δex = %13.6e, Δey = %13.6e, Δi = %12.9f deg, ΔΩ = %12.9f deg, ΔP = %11.6f m, ΔV = %11.9f m/s%n";

    /** Key to print the title for the final frozen eccentricity. */
    private static final String FROZEN_ECC_TITLE = "%nfinal frozen eccentricity%n";

    /** Key to print the the final frozen eccentricity. */
    private static final String FROZEN_ECC = "    ex_f = %17.10e, ey_f = %17.10e%n";

    /** GMST function. */
    private final TimeScalarFunction gmst;

    /** Gravity field. */
    private NormalizedSphericalHarmonicsProvider gravityField;

    /** Earth model. */
    private final BodyShape earth;

    /** Simple constructor.
     */
    public Phasing() {
        final IERSConventions conventions = IERSConventions.IERS_2010;
        final boolean         simpleEOP   = false;
        gmst  = conventions.getGMSTFunction(TimeScalesFactory.getUT1(conventions, simpleEOP));
        earth = ReferenceEllipsoid.getWgs84(FramesFactory.getGTOD(conventions, simpleEOP));
    }

    /** Program entry point.
     * @param args program arguments
     */
    public static void main(final String[] args) {
        try {

            if (args.length != 1) {
                System.err.println("usage: java org.orekit.tutorials.bodies.Phasing resource-name");
                System.exit(1);
            }

            // configure Orekit
            final File home       = new File(System.getProperty("user.home"));
            final File orekitData = new File(home, "orekit-data");
            if (!orekitData.exists()) {
                System.err.format(Locale.US, "Failed to find %s folder%n",
                                  orekitData.getAbsolutePath());
                System.err.format(Locale.US, "You need to download %s from %s, unzip it in %s and rename it 'orekit-data' for this tutorial to work%n",
                                  "orekit-data-master.zip", "https://gitlab.orekit.org/orekit/orekit-data/-/archive/master/orekit-data-master.zip",
                                  home.getAbsolutePath());
                System.exit(1);
            }
            final DataProvidersManager manager = DataContext.getDefault().getDataProvidersManager();
            manager.addProvider(new DirectoryCrawler(orekitData));

            // input/out
            final URL url = Phasing.class.getResource("/" + args[0]);
            if (url == null) {
                System.err.println("resource " + args[0] + " not found");
                System.exit(1);
            }
            final File input  = new File(url.toURI().getPath());

            new Phasing().run(input);

        } catch (URISyntaxException | IOException | IllegalArgumentException | OrekitException e) {
            System.err.println(e.getLocalizedMessage());
            System.exit(1);
        }
    }

    /** run the program.
     * @param input input file
     * @throws IOException if input file cannot be read
     */
    private void run(final File input) throws IOException {

        // read input parameters
        final ObjectMapper mapper = new ObjectMapper(new YAMLFactory());
        mapper.findAndRegisterModules();
        final TutorialPhasing inputData = mapper.readValue(input, TutorialPhasing.class);

        final TimeScale utc = TimeScalesFactory.getUTC();

        // simulation properties
        final AbsoluteDate       date          = new AbsoluteDate(inputData.getOrbit().getDate(), utc);
        final Frame              frame         = inputData.getOrbit().getInertialFrame();
        final int                nbOrbits      = inputData.getPhasingOrbitsNumber();
        final int                nbDays        = inputData.getPhasingDaysNumer();
        final int                maxMDaily     = inputData.getMaxMDaily();
        final double             latitude      = FastMath.toRadians(inputData.getReferenceLatitude());
        final boolean            ascending     = inputData.isReferenceAscending();
        final double             mst           = TimeComponents.parseTime(inputData.getMeanSolarTime()).getSecondsInUTCDay() / 3600;
        final int                degree        = inputData.getGravity().getDegree();
        final int                order         = inputData.getGravity().getOrder();
        final String             objectName    = inputData.getObjectName();
        final String             objectId      = inputData.getObjectId();
        final List<TutorialGrid> grids         = inputData.getGrids();
        final double[]           gridLatitudes = new double[grids.size()];
        final boolean[]          gridAscending = new boolean[grids.size()];
        final double             oemStep       = inputData.getOemStep();
        final double             oemDuration   = inputData.getOemDuration();

        // Fill arrays
        int index = 0;
        for (final TutorialGrid grid : grids) {
            gridLatitudes[index] = FastMath.toRadians(grid.getLatitude());
            gridAscending[index] = grid.isAscending();
            index++;
        }

        gravityField = GravityFieldFactory.getNormalizedProvider(degree, order);

        // initial guess for orbit
        CircularOrbit orbit = guessOrbit(date, frame, nbOrbits, nbDays,
                                         latitude, ascending, mst);
        System.out.format(Locale.US, INITIAL_OSC_ORBIT);
        System.out.format(Locale.US, EPOCH,
                          orbit.getDate());
        System.out.format(Locale.US, ORBITAL_ELEMENTS,
                          orbit.getA(), orbit.getCircularEx(), orbit.getCircularEy(),
                          FastMath.toDegrees(orbit.getI()),
                          FastMath.toDegrees(orbit.getRightAscensionOfAscendingNode()));
        System.out.format(Locale.US, "please wait while orbit is adjusted...%n%n");

        // numerical model for improving orbit
        final OrbitType orbitType =  OrbitType.CIRCULAR;
        final double[][] tolerances = NumericalPropagator.tolerances(0.001, orbit, orbitType);
        final DormandPrince853Integrator integrator =
                new DormandPrince853Integrator(1.0e-5 * orbit.getKeplerianPeriod(),
                                               1.0e-1 * orbit.getKeplerianPeriod(),
                                               tolerances[0], tolerances[1]);
        integrator.setInitialStepSize(1.0e-2 * orbit.getKeplerianPeriod());
        final NumericalPropagator propagator = new NumericalPropagator(integrator);
        propagator.setOrbitType(orbitType);
        propagator.addForceModel(new HolmesFeatherstoneAttractionModel(FramesFactory.getGTOD(IERSConventions.IERS_2010, true),
                                                                       gravityField));
        propagator.addForceModel(new ThirdBodyAttraction(CelestialBodyFactory.getSun()));
        propagator.addForceModel(new ThirdBodyAttraction(CelestialBodyFactory.getMoon()));

        double deltaP = Double.POSITIVE_INFINITY;
        double deltaV = Double.POSITIVE_INFINITY;

        int counter = 0;
        while (deltaP > 1.0 || deltaV > 1.0e-3) {

            final CircularOrbit previous = orbit;

            final CircularOrbit tmp1 = improveEarthPhasing(previous, nbOrbits, nbDays, propagator);
            final CircularOrbit tmp2 = improveSunSynchronization(tmp1, nbOrbits * tmp1.getKeplerianPeriod(),
                                                                 latitude, ascending, mst, propagator);
            orbit = improveFrozenEccentricity(tmp2, propagator, maxMDaily, nbDays, nbOrbits);

            final double da  = orbit.getA() - previous.getA();
            final double dex = orbit.getCircularEx() - previous.getCircularEx();
            final double dey = orbit.getCircularEy() - previous.getCircularEy();
            final double di  = FastMath.toDegrees(orbit.getI() - previous.getI());
            final double dr  = FastMath.toDegrees(orbit.getRightAscensionOfAscendingNode() -
                                           previous.getRightAscensionOfAscendingNode());
            final PVCoordinates delta = new PVCoordinates(previous.getPVCoordinates(),
                                                          orbit.getPVCoordinates());
            deltaP = delta.getPosition().getNorm();
            deltaV = delta.getVelocity().getNorm();

            System.out.format(Locale.US,
                              ITERATION_REPORT,
                              ++counter, da, dex, dey, di, dr, deltaP, deltaV);

        }

        // final orbit
        System.out.format(Locale.US, FINAL_OSC_ORBIT);
        System.out.format(Locale.US, EPOCH,
                          orbit.getDate());
        System.out.format(Locale.US, ORBITAL_ELEMENTS,
                          orbit.getA(), orbit.getCircularEx(), orbit.getCircularEy(),
                          FastMath.toDegrees(orbit.getI()),
                          FastMath.toDegrees(orbit.getRightAscensionOfAscendingNode()));
        System.out.format(Locale.US, FROZEN_ECC_TITLE);
        final MeanEccentricityFitter eccentricityFitter = new MeanEccentricityFitter(orbit, maxMDaily, nbDays, nbOrbits);
        eccentricityFitter.fit(propagator);
        System.out.format(Locale.US, FROZEN_ECC,
                          eccentricityFitter.cx, eccentricityFitter.cy);

        // generate the ground track grid file
        final File gridFile = new File(input.getParent(), objectName + ".grd");
        try (PrintStream output = new PrintStream(gridFile, StandardCharsets.UTF_8.name())) {
            for (int i = 0; i < gridLatitudes.length; ++i) {
                printGridPoints(output, gridLatitudes[i], gridAscending[i], orbit, propagator, nbOrbits);
            }
        }
        System.out.format(Locale.US, GENERATED, gridFile.getAbsolutePath());

        if (oemStep > 0) {
            // generate ephemeris

            // prepare header
            final OdmHeader header = new OdmHeader();
            header.setCreationDate(new AbsoluteDate(LocalDateTime.now(ZoneId.of("Etc/UTC")).toString(), TimeScalesFactory.getUTC()));
            header.setOriginator("CS GROUP");

            // prepare template for metadata
            final OemMetadata metadataTemplate = new OemMetadata(4);
            metadataTemplate.setObjectName(objectName);
            metadataTemplate.setObjectID(objectId);
            metadataTemplate.setCenter(new BodyFacade("EARTH", CelestialBodyFactory.getCelestialBodies().getEarth()));
            metadataTemplate.setReferenceFrame(FrameFacade.map(orbit.getFrame()));
            metadataTemplate.setTimeSystem(TimeSystem.UTC);

            final String oemName = objectName + ".oem";
            final File oemFile = new File(input.getParent(), oemName);
            try (PrintStream  output   = new PrintStream(oemFile, StandardCharsets.UTF_8.name());
                 Generator generator   = new KvnGenerator(output, OemWriter.KVN_PADDING_WIDTH, oemName,
                                                          Constants.JULIAN_DAY, -1);
                 StreamingOemWriter sw = new StreamingOemWriter(generator,
                                                                new WriterBuilder().buildOemWriter(),
                                                                header, metadataTemplate)) {

                // let the propagator generate the ephemeris
                propagator.resetInitialState(new SpacecraftState(orbit));
                propagator.getMultiplexer().clear();
                propagator.getMultiplexer().add(oemStep, sw.newSegment());
                propagator.propagate(orbit.getDate().shiftedBy(oemDuration));

            }
            System.out.format(Locale.US, GENERATED, oemFile.getAbsolutePath());
        }

    }

    /** Fitter for mean eccentricity model.
     * <ul>
     *  <li>the mean model is harmonic at frozen eccentricity pulsation</li>
     *  <li>the short period terms with periods T and T/3 are removed analytically during fitting</li>
     * </ul>
     */
    private class MeanEccentricityFitter {

        /** Initial orbit. */
        private final CircularOrbit initial;

        /** Cycle end date. */
        private final AbsoluteDate tEnd;

        /** Maximum index of m-daily terms. */
        private final int maxMDaily;

        /** Observation points for the fitting. */
        private final List<CircularOrbit> observed;

        /** Frozen eccentricity pulsation. */
        private double eta;

        /** X component of period T harmonic. */
        private double x1;

        /** Y component of period T harmonic. */
        private double y1;

        /** Common X/Y component of period T/3 harmonic. */
        private double xy3;

        /** Center X component of the mean eccentricity. */
        private double cx;

        /** Center Y component of the mean eccentricity. */
        private double cy;

        /** Initial X offset of mean eccentricity. */
        private double dx0;

        /** Initial Y offset of mean eccentricity. */
        private double dy0;

        /** Simple constructor.
         * @param initial orbit at start time
         * @param maxMDaily maximum index of m-daily terms
         * @param nbDays number of days of the phasing cycle
         * @param nbOrbits number of orbits of the phasing cycle
         */
        MeanEccentricityFitter(final CircularOrbit initial, final int maxMDaily,
                               final int nbDays, final int nbOrbits) {
            this.initial   = initial;
            this.tEnd      = initial.getDate().shiftedBy(nbDays * Constants.JULIAN_DAY);
            this.maxMDaily = maxMDaily;
            this.observed  = new ArrayList<>();
        }

        /** Perform fitting.
         * @param propagator propagator to use
         */
        public void fit(final Propagator propagator) {

            propagator.resetInitialState(new SpacecraftState(initial));
            final AbsoluteDate start = initial.getDate();

            // sample orbit for one phasing cycle
            propagator.getMultiplexer().clear();
            propagator.getMultiplexer().add(60, state -> observed.add((CircularOrbit) OrbitType.CIRCULAR.convertType(state.getOrbit())));
            propagator.propagate(start, tEnd);

            // compute mean semi-major axis and mean inclination
            final double meanA = observed.stream().collect(Collectors.averagingDouble(c -> c.getA()));
            final double meanI = observed.stream().collect(Collectors.averagingDouble(c -> c.getI()));

            // extract gravity field data
            final double referenceRadius     = gravityField.getAe();
            final double mu                  = gravityField.getMu();
            final double[][] unnormalization = GravityFieldFactory.getUnnormalizationFactors(2, 0);
            final double j2                  = -unnormalization[2][0] * gravityField.onDate(initial.getDate()).getNormalizedCnm(2, 0);

            // compute coefficient of harmonic terms
            final double meanMotion = FastMath.sqrt(mu / meanA) / meanA;
            final double sinI       = FastMath.sin(meanI);
            final double sinI2      = sinI * sinI;
            final double rOa        = referenceRadius / meanA;

            // mean frozen eccentricity pulsation (long period)
            this.eta                = 3 * meanMotion * j2 * rOa * rOa * (1.25 * sinI2 - 1.0);

            // short periods
            final double kappa1 = 1.5 * j2 * rOa * rOa;
            final double kappa2 = kappa1 / (1.0 - kappa1 * (3.0 - 4.0 * sinI2));
            x1  = kappa2 * (1.0 - 1.25 * sinI2);
            y1  = kappa2 * (1.0 - 1.75 * sinI2);
            xy3 = kappa2 * 7.0 / 12.0 * sinI2;

            final LeastSquaresProblem lsp = new LeastSquaresBuilder().
                                            maxEvaluations(1000).
                                            maxIterations(1000).
                                            start(new double[4 + 4 * maxMDaily]).
                                            target(new double[2 * observed.size()]).
                                            model(params -> residuals(params),
                                                params -> jacobian(params)).build();
            final LeastSquaresOptimizer.Optimum optimum = new LevenbergMarquardtOptimizer().optimize(lsp);

            // store coefficients (for mean model only)
            cx  = optimum.getPoint().getEntry(0);
            cy  = optimum.getPoint().getEntry(1);
            dx0 = optimum.getPoint().getEntry(2);
            dy0 = optimum.getPoint().getEntry(3);

        }

        /** Value of the error model.
         * @param params fitting parameters
         * @return model value
         */
        private double[] residuals(final double[] params) {
            final double[] val = new double[2 * observed.size()];
            int i = 0;
            for (final CircularOrbit c : observed) {
                final double dt     = c.getDate().durationFrom(initial.getDate());
                final double alphaM = c.getAlphaM();
                final SinCos sc     = FastMath.sinCos(eta * dt);
                final SinCos sc1    = FastMath.sinCos(alphaM);
                final SinCos sc3    = FastMath.sinCos(3 * alphaM);
                final SinCos[] scM  = new SinCos[maxMDaily];
                for (int k = 0; k < scM.length; ++k) {
                    scM[k] = FastMath.sinCos(2 * (k + 1) * FastMath.PI * dt / Constants.JULIAN_DAY);
                };
                final double exM  = params[0] + params[2] * sc.cos()  + params[3] * sc.sin();
                final double eyM  = params[1] - params[2] * sc.sin()  + params[3] * sc.cos();
                final double exJ2 = x1 * sc1.cos() + xy3 * sc3.cos();
                final double eyJ2 = y1 * sc1.sin() + xy3 * sc3.sin();
                double exS = 0;
                double eyS = 0;
                for (int k = 0; k < scM.length; ++k) {
                    exS += params[4 * k + 4] * scM[k].cos() + params[4 * k + 5] * scM[k].sin();
                    eyS += params[4 * k + 6] * scM[k].cos() + params[4 * k + 7] * scM[k].sin();
                }
                val[i++] = exM + exJ2 + exS - c.getCircularEx();
                val[i++] = eyM + eyJ2 + eyS - c.getCircularEy();
            }
            return val;
        }

        /** Jacobian of the error model.
         * @param params fitting parameters
         * @return model Jacobian
         */
        private double[][] jacobian(final double[] params) {
            final double[][] jac = new double[2 * observed.size()][];
            int i = 0;
            for (final CircularOrbit c : observed) {
                final double dt = c.getDate().durationFrom(initial.getDate());
                final SinCos sc = FastMath.sinCos(eta * dt);
                final SinCos[] scM  = new SinCos[maxMDaily];
                for (int k = 0; k < scM.length; ++k) {
                    scM[k] = FastMath.sinCos(2 * (k + 1) * FastMath.PI * dt / Constants.JULIAN_DAY);
                };
                final double[] jacX = new double[4 + 4 * scM.length];
                final double[] jacY = new double[4 + 4 * scM.length];
                jacX[0] = 1;
                jacX[1] = 0;
                jacX[2] = sc.cos();
                jacX[3] = sc.sin();
                jacY[0] = 0;
                jacY[1] = 1;
                jacY[2] = -sc.sin();
                jacY[3] = sc.cos();
                for (int k = 0; k < scM.length; ++k) {
                    jacX[4 + 4 * k] = scM[k].cos();
                    jacX[5 + 4 * k] = scM[k].sin();
                    jacY[6 + 4 * k] = scM[k].cos();
                    jacY[7 + 4 * k] = scM[k].sin();
                }
                jac[i++] = jacX;
                jac[i++] = jacY;
            }
            return jac;
        }

    }

    /** Guess an initial orbit from theoretical model.
     * @param date orbit date
     * @param frame frame to use for defining orbit
     * @param nbOrbits number of orbits in the phasing cycle
     * @param nbDays number of days in the phasing cycle
     * @param latitude reference latitude for Sun synchronous orbit
     * @param ascending if true, crossing latitude is from South to North
     * @param mst desired mean solar time at reference latitude crossing
     * @return an initial guess of Earth phased, Sun synchronous orbit
     */
    private CircularOrbit guessOrbit(final AbsoluteDate date, final Frame frame,
                                     final int nbOrbits, final int nbDays,
                                     final double latitude, final boolean ascending,
                                     final double mst) {

        final double mu = gravityField.getMu();
        final NormalizedSphericalHarmonicsProvider.NormalizedSphericalHarmonics harmonics =
                gravityField.onDate(date);

        // initial semi major axis guess based on Keplerian period
        final double period0 = (nbDays * Constants.JULIAN_DAY) / nbOrbits;
        final double n0      = 2 * FastMath.PI / period0;
        final double a0      = FastMath.cbrt(mu / (n0 * n0));

        // initial inclination guess based on ascending node drift due to J2
        final double[][] unnormalization = GravityFieldFactory.getUnnormalizationFactors(3, 0);
        final double j2       = -unnormalization[2][0] * harmonics.getNormalizedCnm(2, 0);
        final double j3       = -unnormalization[3][0] * harmonics.getNormalizedCnm(3, 0);
        final double raanRate = 2 * FastMath.PI / Constants.JULIAN_YEAR;
        final double ae       = gravityField.getAe();
        final double i0       = FastMath.acos(-raanRate * a0 * a0 / (1.5 * ae * ae * j2 * n0));

        // initial eccentricity guess based on J2 and J3
        final double ex0   = 0;
        final double ey0   = -j3 * ae * FastMath.sin(i0) / (2 * a0 * j2);

        // initial ascending node guess based on mean solar time
        double alpha0 = FastMath.asin(FastMath.sin(latitude) / FastMath.sin(i0));
        if (!ascending) {
            alpha0 = FastMath.PI - alpha0;
        }
        final double h = meanSolarTime(new CircularOrbit(a0, ex0, ey0, i0, 0.0, alpha0,
                                                         PositionAngleType.TRUE, frame, date, mu));
        final double raan0 = FastMath.PI * (mst - h) / 12.0;

        return new CircularOrbit(a0, ex0, ey0, i0, raan0, alpha0,
                                 PositionAngleType.TRUE, frame, date, mu);

    }

    /** Improve orbit to better match Earth phasing parameters.
     * @param previous previous orbit
     * @param nbOrbits number of orbits in the phasing cycle
     * @param nbDays number of days in the phasing cycle
     * @param propagator propagator to use
     * @return an improved Earth phased orbit
     */
    private CircularOrbit improveEarthPhasing(final CircularOrbit previous, final int nbOrbits, final int nbDays,
                                              final Propagator propagator) {

        propagator.resetInitialState(new SpacecraftState(previous));

        // find first ascending node
        double period = previous.getKeplerianPeriod();
        final SpacecraftState firstState = findFirstCrossing(0.0, true, previous.getDate(),
                                                             previous.getDate().shiftedBy(2 * period),
                                                             0.01 * period, propagator);

        // go to next cycle, one orbit at a time
        SpacecraftState state = firstState;
        for (int i = 0; i < nbOrbits; ++i) {
            final AbsoluteDate previousDate = state.getDate();
            state = findLatitudeCrossing(0.0, previousDate.shiftedBy(period),
                                         previousDate.shiftedBy(2 * period),
                                         0.01 * period, period, propagator);
            period = state.getDate().durationFrom(previousDate);
        }

        final double cycleDuration = state.getDate().durationFrom(firstState.getDate());
        final double deltaT;
        if (((int) FastMath.rint(cycleDuration / Constants.JULIAN_DAY)) != nbDays) {
            // we are very far from expected duration
            deltaT = nbDays * Constants.JULIAN_DAY - cycleDuration;
        } else {
            // we are close to expected duration
            final GeodeticPoint startPoint = earth.transform(firstState.getPVCoordinates().getPosition(),
                                                             firstState.getFrame(), firstState.getDate());
            final GeodeticPoint endPoint   = earth.transform(state.getPVCoordinates().getPosition(),
                                                             state.getFrame(), state.getDate());
            final double deltaL =
                    MathUtils.normalizeAngle(endPoint.getLongitude() - startPoint.getLongitude(), 0.0);
            deltaT = deltaL * Constants.JULIAN_DAY / (2 * FastMath.PI);
        }

        final double deltaA = 2 * previous.getA() * deltaT / (3 * nbOrbits * previous.getKeplerianPeriod());
        return new CircularOrbit(previous.getA() + deltaA,
                                 previous.getCircularEx(), previous.getCircularEy(),
                                 previous.getI(), previous.getRightAscensionOfAscendingNode(),
                                 previous.getAlphaV(), PositionAngleType.TRUE,
                                 previous.getFrame(), previous.getDate(),
                                 previous.getMu());

    }

    /** Improve orbit to better match phasing parameters.
     * @param previous previous orbit
     * @param duration sampling duration
     * @param latitude reference latitude for Sun synchronous orbit
     * @param ascending if true, crossing latitude is from South to North
     * @param mst desired mean solar time at reference latitude crossing
     * @param propagator propagator to use
     * @return an improved Earth phased, Sun synchronous orbit
     */
    private CircularOrbit improveSunSynchronization(final CircularOrbit previous, final double duration,
                                                    final double latitude, final boolean ascending, final double mst,
                                                    final Propagator propagator) {

        propagator.resetInitialState(new SpacecraftState(previous));
        final AbsoluteDate start = previous.getDate();

        // find the first latitude crossing
        double period   = previous.getKeplerianPeriod();
        final double stepSize = period / 100;
        SpacecraftState crossing =
                findFirstCrossing(latitude, ascending, start, start.shiftedBy(2 * period),
                                  stepSize, propagator);

        // find all other latitude crossings from regular schedule
        final SecularAndHarmonic mstModel = new SecularAndHarmonic(2,
                                                                   2.0 * FastMath.PI / Constants.JULIAN_YEAR,
                                                                   4.0 * FastMath.PI / Constants.JULIAN_YEAR,
                                                                   2.0 * FastMath.PI / Constants.JULIAN_DAY,
                                                                   4.0 * FastMath.PI / Constants.JULIAN_DAY);
        mstModel.resetFitting(start, new double[] {
            mst, -1.0e-10, -1.0e-17,
            1.0e-3, 1.0e-3, 1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5, 1.0e-5
        });
        while (crossing != null && crossing.getDate().durationFrom(start) < duration) {
            final AbsoluteDate previousDate = crossing.getDate();
            crossing = findLatitudeCrossing(latitude, previousDate.shiftedBy(period),
                                            previousDate.shiftedBy(2 * period),
                                            stepSize, period / 8, propagator);
            if (crossing != null) {

                // store current point
                mstModel.addPoint(crossing.getDate(), meanSolarTime(crossing.getOrbit()));

                // use the same time separation to pinpoint next crossing
                period = crossing.getDate().durationFrom(previousDate);

            }

        }

        // fit the mean solar time to a parabolic plus medium periods model
        // we will only use the linear part for the correction
        mstModel.fit();
        final double[] fittedH = mstModel.approximateAsPolynomialOnly(1, start, 2, 2,
                                                                      start, start.shiftedBy(duration),
                                                                      stepSize);

        // solar time bias must be compensated by shifting ascending node
        final double deltaRaan = FastMath.PI * (mst - fittedH[0]) / 12;

        // solar time slope must be compensated by changing inclination
        // linearized relationship between hDot and inclination:
        // hDot = alphaDot - raanDot where alphaDot is the angular rate of Sun right ascension
        // and raanDot is the angular rate of ascending node right ascension. So hDot evolution
        // is the opposite of raan evolution, which itself is proportional to cos(i) due to J2
        // effect. So hDot = alphaDot - k cos(i) and hence Delta hDot = -k sin(i) Delta i
        // so Delta hDot / Delta i = (alphaDot - hDot) tan(i)
        final double dhDotDi = (24.0 / Constants.JULIAN_YEAR - fittedH[1]) * FastMath.tan(previous.getI());

        // compute inclination offset needed to achieve station-keeping target
        final double deltaI = fittedH[1] / dhDotDi;

        return new CircularOrbit(previous.getA(),
                                 previous.getCircularEx(), previous.getCircularEy(),
                                 previous.getI() + deltaI,
                                 previous.getRightAscensionOfAscendingNode() + deltaRaan,
                                 previous.getAlphaV(), PositionAngleType.TRUE,
                                 previous.getFrame(), previous.getDate(),
                                 previous.getMu());

    }

    /** Fit eccentricity model.
     * @param previous previous orbit
     * @param propagator propagator to use
     * @param maxMDaily maximum index of m-daily terms
     * @param nbDays number of days of the phasing cycle
     * @param nbOrbits number of orbits of the phasing cycle
     * @return orbit with improved frozen eccentricity
     */
    private CircularOrbit improveFrozenEccentricity(final CircularOrbit previous, final Propagator propagator,
                                                    final int maxMDaily, final int nbDays, final int nbOrbits) {

        // fit mean eccentricity over one phasing cycle
        final MeanEccentricityFitter eccentricityFitter = new MeanEccentricityFitter(previous, maxMDaily, nbDays, nbOrbits);
        eccentricityFitter.fit(propagator);

        // collapse mean evolution circular motion to its center point
        return new CircularOrbit(previous.getA(),
                                 previous.getCircularEx() - eccentricityFitter.dx0,
                                 previous.getCircularEy() - eccentricityFitter.dy0,
                                 previous.getI(), previous.getRightAscensionOfAscendingNode(),
                                 previous.getAlphaV(), PositionAngleType.TRUE,
                                 previous.getFrame(), previous.getDate(),
                                 previous.getMu());


    }

    /** Print ground track grid point.
     * @param out output stream
     * @param latitude point latitude
     * @param ascending indicator for latitude crossing direction
     * @param orbit phased orbit
     * @param propagator propagator for orbit
     * @param nbOrbits number of orbits in the cycle
     */
    private void printGridPoints(final PrintStream out,
                                 final double latitude, final boolean ascending,
                                 final Orbit orbit, final Propagator propagator, final int nbOrbits) {

        propagator.resetInitialState(new SpacecraftState(orbit));
        final AbsoluteDate start = orbit.getDate();

        // find the first latitude crossing
        double period   = orbit.getKeplerianPeriod();
        final double stepSize = period / 100;
        SpacecraftState crossing =
                findFirstCrossing(latitude, ascending, start, start.shiftedBy(2 * period),
                                  stepSize, propagator);

        // find all other latitude crossings from regular schedule
        for (int i = 0; i < nbOrbits; ++i) {

            final CircularOrbit c = (CircularOrbit) OrbitType.CIRCULAR.convertType(crossing.getOrbit());
            final GeodeticPoint gp = earth.transform(crossing.getPVCoordinates().getPosition(),
                                                     crossing.getFrame(), crossing.getDate());
            out.format(Locale.US, "%11.3f %9.5f %9.5f %s %11.5f%n",
                       crossing.getDate().durationFrom(start),
                       FastMath.toDegrees(gp.getLatitude()), FastMath.toDegrees(gp.getLongitude()),
                       ascending,
                       FastMath.toDegrees(MathUtils.normalizeAngle(c.getAlphaV(), 0)));

            final AbsoluteDate previousDate = crossing.getDate();
            crossing = findLatitudeCrossing(latitude, previousDate.shiftedBy(period),
                                            previousDate.shiftedBy(2 * period),
                                            stepSize, period / 8, propagator);
            period = crossing.getDate().durationFrom(previousDate);

        }

    }

    /** Compute the mean solar time.
     * @param orbit current orbit
     * @return mean solar time
     */
    private double meanSolarTime(final Orbit orbit) {

        // compute angle between Sun and spacecraft in the equatorial plane
        final Vector3D position = orbit.getPVCoordinates().getPosition();
        final double time       = orbit.getDate().getComponents(TimeScalesFactory.getUTC()).getTime().getSecondsInUTCDay();
        final double theta      = gmst.value(orbit.getDate());
        final double sunAlpha   = theta + FastMath.PI * (1 - time / (Constants.JULIAN_DAY * 0.5));
        final double dAlpha     = MathUtils.normalizeAngle(position.getAlpha() - sunAlpha, 0);

        // convert the angle to solar time
        return 12.0 * (1.0 + dAlpha / FastMath.PI);

    }

    /**
     * Find the first crossing of the reference latitude.
     * @param latitude latitude to search for
     * @param ascending indicator for desired crossing direction
     * @param searchStart search start
     * @param end maximal date not to overtake
     * @param stepSize step size to use
     * @param propagator propagator
     * @return first crossing
     */
    private SpacecraftState findFirstCrossing(final double latitude, final boolean ascending,
                                              final AbsoluteDate searchStart, final AbsoluteDate end,
                                              final double stepSize, final Propagator propagator) {

        double previousLatitude = Double.NaN;
        for (AbsoluteDate date = searchStart; date.compareTo(end) < 0; date = date.shiftedBy(stepSize)) {
            final PVCoordinates pv       = propagator.propagate(date).getPVCoordinates(earth.getBodyFrame());
            final double currentLatitude = earth.transform(pv.getPosition(), earth.getBodyFrame(), date).getLatitude();
            if (previousLatitude <= latitude && currentLatitude >= latitude &&  ascending ||
                previousLatitude >= latitude && currentLatitude <= latitude && !ascending) {
                return findLatitudeCrossing(latitude, date.shiftedBy(-0.5 * stepSize), end,
                                            0.5 * stepSize, 2 * stepSize, propagator);
            }
            previousLatitude = currentLatitude;
        }

        throw new OrekitException(LocalizedCoreFormats.SIMPLE_MESSAGE,
                                  "latitude " + FastMath.toDegrees(latitude) + " never crossed");

    }


    /**
     * Find the state at which the reference latitude is crossed.
     * @param latitude latitude to search for
     * @param guessDate guess date for the crossing
     * @param endDate maximal date not to overtake
     * @param shift shift value used to evaluate the latitude function bracketing around the guess date
     * @param maxShift maximum value that the shift value can take
     * @param propagator propagator used
     * @return state at latitude crossing time
     * @throws MathRuntimeException if latitude cannot be bracketed in the search interval
     */
    private SpacecraftState findLatitudeCrossing(final double latitude,
                                                 final AbsoluteDate guessDate, final AbsoluteDate endDate,
                                                 final double shift, final double maxShift,
                                                 final Propagator propagator)
        throws MathRuntimeException {

        // function evaluating to 0 at latitude crossings
        final UnivariateFunction latitudeFunction = new UnivariateFunction() {
            /** {@inheritDoc} */
            public double value(final double x) {
                try {
                    final SpacecraftState state = propagator.propagate(guessDate.shiftedBy(x));
                    final Vector3D position = state.getPVCoordinates(earth.getBodyFrame()).getPosition();
                    final GeodeticPoint point = earth.transform(position, earth.getBodyFrame(), state.getDate());
                    return point.getLatitude() - latitude;
                } catch (OrekitException oe) {
                    throw new RuntimeException(oe);
                }
            }
        };

        // try to bracket the encounter
        double span;
        if (guessDate.shiftedBy(shift).compareTo(endDate) > 0) {
            // Take a 1e-3 security margin
            span = endDate.durationFrom(guessDate) - 1e-3;
        } else {
            span = shift;
        }

        while (!UnivariateSolverUtils.isBracketing(latitudeFunction, -span, span)) {

            if (2 * span > maxShift) {
                // let the Hipparchus exception be thrown
                UnivariateSolverUtils.verifyBracketing(latitudeFunction, -span, span);
            } else if (guessDate.shiftedBy(2 * span).compareTo(endDate) > 0) {
                // Out of range :
                return null;
            }

            // expand the search interval
            span *= 2;

        }

        // find the encounter in the bracketed interval
        final BaseUnivariateSolver<UnivariateFunction> solver =
                new BracketingNthOrderBrentSolver(0.1, 5);
        final double dt = solver.solve(1000, latitudeFunction, -span, span);
        return propagator.propagate(guessDate.shiftedBy(dt));

    }

    /**
     * Input data for the Phasing tutorial.
     * <p>
     * Data are read from a YAML file.
     * </p>
     * @author Bryan Cazabonne
     */
    public static class TutorialPhasing {

        /** Name of the object. */
        private String objectName;

        /** Id of the object. */
        private String objectId;

        /** Orbit data. */
        private TutorialOrbit orbit;

        /** Number of orbits in the phasing cycle. */
        private int phasingOrbitsNumber;

        /** Number of days in the phasing cycle. */
        private int phasingDaysNumer;

        /** Reference latitude for Sun synchronous orbit (°). */
        private double referenceLatitude;

        /** Reference indicator for latitude crossing direction. */
        private boolean referenceAscending;

        /** Mean solar time. */
        private String meanSolarTime;

        /** Gravity data. */
        private TutorialGravity gravity;

        /** Maximum order of m-daily terms for eccentricity fitting. */
        private int maxMDaily;

        /** List of grid data. */
        private List<TutorialGrid> grids;

        /** OEM step size. */
        private double oemStep;

        /** OEM duration. */
        private double oemDuration;

        /**
         * Get the name of the object.
         * @return the name of the object.
         */
        public String getObjectName() {
            return objectName;
        }

        /**
         * Set the name of the object.
         * @param objectName name of the ground track grid file
         */
        public void setObjectName(final String objectName) {
            this.objectName = objectName;
        }

        /**
         * Get the id of the object.
         * @return id of the object.
         */
        public String getObjectId() {
            return objectId;
        }

        /**
         * Set the id of the object.
         * @param objectId id of the object
         */
        public void setObjectId(final String objectId) {
            this.objectId = objectId;
        }

        /**
         * Get the orbit data.
         * @return the orbit data
         */
        public TutorialOrbit getOrbit() {
            return orbit;
        }

        /**
         * Set the orbit data.
         * @param orbit orbit data
         */
        public void setOrbit(final TutorialOrbit orbit) {
            this.orbit = orbit;
        }

        /**
         * Get the number of orbits in the phasing cycle.
         * @return the number of orbits in the phasing cycle
         */
        public int getPhasingOrbitsNumber() {
            return phasingOrbitsNumber;
        }

        /**
         * Set the number of orbits in the phasing cycle.
         * @param phasingOrbitsNumber number of orbits in the phasing cycle
         */
        public void setPhasingOrbitsNumber(final int phasingOrbitsNumber) {
            this.phasingOrbitsNumber = phasingOrbitsNumber;
        }

        /**
         * Get the number of days in the phasing cycle.
         * @return the number of days in the phasing cycle.
         */
        public int getPhasingDaysNumer() {
            return phasingDaysNumer;
        }

        /**
         * Set the number of days in the phasing cycle.
         * @param phasingDaysNumer number of days in the phasing cycle
         */
        public void setPhasingDaysNumer(final int phasingDaysNumer) {
            this.phasingDaysNumer = phasingDaysNumer;
        }

        /**
         * Get the reference latitude for Sun synchronous orbit.
         * @return the reference latitude for Sun synchronous orbit (°)
         */
        public double getReferenceLatitude() {
            return referenceLatitude;
        }

        /**
         * Set the reference latitude for Sun synchronous orbit.
         * @param referenceLatitude reference latitude for Sun synchronous orbit (°)
         */
        public void setReferenceLatitude(final double referenceLatitude) {
            this.referenceLatitude = referenceLatitude;
        }

        /**
         * Get the reference indicator for latitude crossing direction.
         * @return if true, crossing latitude is from South to North
         */
        public boolean isReferenceAscending() {
            return referenceAscending;
        }

        /**
         * Set the reference indicator for latitude crossing direction.
         * @param referenceAscending if true, crossing latitude is from South to North
         */
        public void setReferenceAscending(final boolean referenceAscending) {
            this.referenceAscending = referenceAscending;
        }

        /**
         * Get the string representation of the mean solar time.
         * @return the string representation of the mean solar time
         */
        public String getMeanSolarTime() {
            return meanSolarTime;
        }

        /**
         * Set the string representation of the mean solar time.
         * @param meanSolarTime string representation of the mean solar time
         */
        public void setMeanSolarTime(final String meanSolarTime) {
            this.meanSolarTime = meanSolarTime;
        }

        /**
         * Get the gravity data.
         * @return the gravity data
         */
        public TutorialGravity getGravity() {
            return gravity;
        }

        /**
         * Set the gravity data.
         * @param gravity gravity data
         */
        public void setGravity(final TutorialGravity gravity) {
            this.gravity = gravity;
        }

        /**
         * Set the maximum order of m-daily terms for eccentricity fitting.
         * @param maxMDaily maximum order of m-daily terms for eccentricity fitting
         */
        public void setMaxMDaily(final int maxMDaily) {
            this.maxMDaily = maxMDaily;
        }

        /**
         * Get the maximum order of m-daily terms for eccentricity fitting.
         * @return maximum order of m-daily terms for eccentricity fitting
         */
        public int getMaxMDaily() {
            return maxMDaily;
        }

        /**
         * Get the list of grid data.
         * @return the list of grid data
         */
        public List<TutorialGrid> getGrids() {
            return grids;
        }

        /**
         * Set the list of grid data.
         * @param grids list of grid data
         */
        public void setGrids(final List<TutorialGrid> grids) {
            this.grids = grids;
        }

        /**
         * Set the OEM step size.
         * @param oemStep OEM step size
         */
        public void setOemStep(final double oemStep) {
            this.oemStep = oemStep;
        }

        /**
         * Get the OEM step size.
         * @return OEM step size
         */
        public double getOemStep() {
            return oemStep;
        }

        /**
         * Set the OEM duration.
         * @param oemDuration OEM duration
         */
        public void setOemDuration(final double oemDuration) {
            this.oemDuration = oemDuration;
        }

        /**
         * Get the OEM duration.
         * @return OEM duration
         */
        public double getOemDuration() {
            return oemDuration;
        }

    }

    /**
     * Input data for the grid parameter.
     * <p>
     * Data are read from a YAML file.
     * </p>
     * @author Bryan Cazabonne
     */
    public static class TutorialGrid {

        /** Point latitude (°). */
        private double latitude;

        /** Indicator for latitude crossing direction. */
        private boolean ascending;

        /**
         * Get the point latitude.
         * @return the point latitude (°)
         */
        public double getLatitude() {
            return latitude;
        }

        /**
         * Set the point latitude.
         * @param latitude point latitude (°)
         */
        public void setLatitude(final double latitude) {
            this.latitude = latitude;
        }

        /**
         * Get the indicator for latitude crossing direction.
         * @return if true, crossing latitude is from South to North
         */
        public boolean isAscending() {
            return ascending;
        }

        /**
         * Set the indicator for latitude crossing direction.
         * @param ascending if true, crossing latitude is from South to North
         */
        public void setAscending(final boolean ascending) {
            this.ascending = ascending;
        }

    }

}
