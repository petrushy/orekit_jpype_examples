"""This is a Python translation of the java NodeDetectorTest.java that is part of the orekit test suite. 

/* Copyright 2002-2013 CS mes d'Information (CS) under one or more
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

Modifications and translation to python by Petrus Hyvonen, SSC 2014

"""


import orekit
orekit.initVM()

from orekit import JArray_double
from orekit.pyhelpers import setup_orekit_curdir
from math import radians

from org.orekit.propagation.events import EventsLogger
from org.orekit.propagation.events import NodeDetector

from org.hipparchus.ode.nonstiff import DormandPrince853Integrator;
from org.orekit.frames import FramesFactory;
from org.orekit.orbits import KeplerianOrbit;
from org.orekit.orbits import PositionAngle;
from org.orekit.propagation import SpacecraftState;
from org.orekit.propagation.events.handlers import ContinueOnEvent;
from org.orekit.propagation.numerical import NumericalPropagator;
from org.orekit.time import AbsoluteDate;
from org.orekit.time import TimeScalesFactory;
from org.orekit.utils import Constants;

setup_orekit_curdir()   # orekit-data.zip shall be in current dir

# Floats are needed to be specific in the orekit interface
a = 800000.0 + Constants.WGS84_EARTH_EQUATORIAL_RADIUS;
e = 0.0001;
i = radians(98.0);
w = -90.0;
raan = 0.0;
v = 0.0;

inertialFrame = FramesFactory.getEME2000();
initialDate = AbsoluteDate(2014, 01, 01, 0, 0, 0.0, TimeScalesFactory.getUTC());
finalDate = initialDate.shiftedBy(5000.0);
initialOrbit = KeplerianOrbit(a, e, i, w, raan, v, PositionAngle.TRUE, inertialFrame, initialDate, Constants.WGS84_EARTH_MU);
initialState = SpacecraftState(initialOrbit, 1000.0);

tol = NumericalPropagator.tolerances(10.0, initialOrbit, initialOrbit.getType());

# Double array of doubles needs to be retyped to work
integrator = DormandPrince853Integrator(0.001, 1000.0, 
    JArray_double.cast_(tol[0]),
    JArray_double.cast_(tol[1]))

propagator = NumericalPropagator(integrator);
propagator.setInitialState(initialState);

# Define 2 instances of NodeDetector:
rawDetector = NodeDetector(1e-6, 
        initialState.getOrbit(), 
        initialState.getFrame()).withHandler(ContinueOnEvent().of_(NodeDetector))
        # Converted withHandler(ContinueOnEvent<NodeDetector>());

logger1 = EventsLogger();
node1 = logger1.monitorDetector(rawDetector);
logger2 = EventsLogger();
node2 = logger2.monitorDetector(rawDetector);

propagator.addEventDetector(node1);
propagator.addEventDetector(node2);

# First propagation
propagator.setEphemerisMode();
propagator.propagate(finalDate);

assert 2==logger1.getLoggedEvents().size()
assert 2== logger2.getLoggedEvents().size();

logger1.clearLoggedEvents();
logger2.clearLoggedEvents();

postpro = propagator.getGeneratedEphemeris();

# Post-processing
postpro.addEventDetector(node1);
postpro.addEventDetector(node2);
postpro.propagate(finalDate);

assert 2==logger1.getLoggedEvents().size()
assert 2==logger2.getLoggedEvents().size()


